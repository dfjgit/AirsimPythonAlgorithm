using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 中心人员控制脚本：负责Z形状巡逻矩阵移动、公开关键参数，适配无人机集群仿真需求
/// </summary>
public class LeaderController : MonoBehaviour
{
    public OptimizedHexGrid grid;
    public Color PathColor = Color.green;

    #region 公开配置参数（对应需求，支持Inspector面板调整）

    /// <summary>
    /// 中心人员移动速度（单位：m/s），需求明确“远小于无人机0.5m/s”，默认0.05m/s
    /// </summary>
    [Header("移动配置")]
    [Tooltip("移动速度（m/s），需远小于无人机速度（0.5m/s）")]
    public float moveSpeed = 0.05f;

    /// <summary>
    /// 扫描半径（单位：m），需求明确默认3m，支持配置
    /// </summary>
    [Header("扫描范围配置")]
    [Tooltip("扫描半径（m），无人机仅能在该范围内活动")] 
    public float scanRadius = 3f;

    /// <summary>
    /// Z形巡逻矩阵参数
    /// </summary>
    [Header("Z形巡逻矩阵配置")]
    [Tooltip("巡逻区域起点（左下角）")]
    public Vector3 patrolStartPoint = new Vector3(0, 0.5f, 0);
    
    [Tooltip("巡逻区域宽度（X轴方向）")]
    public float patrolWidth = 10f;
    
    [Tooltip("巡逻区域高度（Z轴方向）")]
    public float patrolHeight = 10f;
    
    [Tooltip("巡逻行数")]
    public int rowCount = 5;
    
    [Tooltip("是否自动生成Z形轨迹")]
    public bool autoGenerateZPattern = true;

    /// <summary>
    /// 固定移动轨迹点列表（世界坐标），如果自动生成Z形轨迹则会忽略手动设置的点
    /// </summary>
    [Header("轨迹配置")]
    [Tooltip("固定轨迹点（按移动顺序排列，Y轴建议统一）")] 
    public List<Vector3> trajectoryPoints = new List<Vector3>
    {
        new Vector3(0, 0.5f, 0),  
        new Vector3(10, 0.5f, 0),
        new Vector3(10, 0.5f, 2),
        new Vector3(0, 0.5f, 2),
        new Vector3(0, 0.5f, 4),
        new Vector3(10, 0.5f, 4)
    };

    #endregion

    #region 公开状态参数（供其他脚本访问，如无人机力向量计算）

    /// <summary>
    /// 中心人员当前世界位置（实时更新）
    /// </summary>
    public Vector3 CurrentPosition { get; private set; }

    /// <summary>
    /// 任务是否结束（对于循环轨迹，此值始终为false）
    /// </summary>
    public bool IsTaskFinished { get; private set; }

    /// <summary>
    /// 当前目标轨迹点索引（用于调试和扩展）
    /// </summary>
    public int CurrentTargetIndex { get; private set; }

    #endregion

    #region 私有变量

    /// <summary>
    /// 当前正在朝向的目标轨迹点
    /// </summary>
    private Vector3 _currentTargetPoint;

    /// <summary>
    /// 判定“到达目标点”的距离阈值（避免因浮点误差无法触发）
    /// </summary>
    private const float _arriveThreshold = 0.1f;

    #endregion

    void Start()
    {
        // 如果启用自动生成Z形轨迹，则生成轨迹点
        if (autoGenerateZPattern)
        {
            GenerateZPatternTrajectory();
        }

        // 初始化：检查轨迹点合法性
        if (trajectoryPoints == null || trajectoryPoints.Count < 2)
        {
            Debug.LogError("【中心人员】轨迹点配置错误！需至少设置2个点（起点+终点）");
            enabled = false; // 禁用脚本，避免异常
            return;
        }

        // 初始化位置和目标点
        CurrentTargetIndex = 0;
        _currentTargetPoint = trajectoryPoints[CurrentTargetIndex];
        CurrentPosition = _currentTargetPoint; // 起点位置
        transform.position = CurrentPosition;  // 同步游戏对象位置

        IsTaskFinished = false;
        Debug.Log($"【中心人员】初始化完成！起点：{CurrentPosition}，扫描半径：{scanRadius}m，速度：{moveSpeed}m/s");
    }

    void Update()
    {
        grid.ScanAreaIncrease(transform.position, scanRadius);
        
        // 任务结束时，停止移动（对于循环轨迹，此条件通常不成立）
        if (IsTaskFinished) return;

        // 1. 移动逻辑：朝向当前目标点移动
        CurrentPosition = Vector3.MoveTowards(
            current: CurrentPosition,
            target: _currentTargetPoint,
            maxDistanceDelta: moveSpeed * Time.deltaTime
        );
        // 同步游戏对象位置到当前位置
        transform.position = CurrentPosition;

        // 2. 检查是否到达当前目标点
        if (Vector3.Distance(CurrentPosition, _currentTargetPoint) <= _arriveThreshold)
        {
            Debug.Log($"【中心人员】到达轨迹点 {CurrentTargetIndex + 1}/{trajectoryPoints.Count}：{_currentTargetPoint}");

            // 3. 切换到下一个目标点，循环回到起点
            CurrentTargetIndex++;
            if (CurrentTargetIndex >= trajectoryPoints.Count)
            {
                // 到达最后一个点，循环回到起点
                CurrentTargetIndex = 0;
                Debug.Log($"【中心人员】完成一轮巡逻，回到起点重新开始");
            }
            
            _currentTargetPoint = trajectoryPoints[CurrentTargetIndex];
        }
    }

    /// <summary>
    /// 生成Z形巡逻轨迹点
    /// </summary>
    private void GenerateZPatternTrajectory()
    {
        trajectoryPoints.Clear();
        
        if (rowCount < 2)
        {
            rowCount = 2;
            Debug.LogWarning("【中心人员】行数不能小于2，已自动调整为2");
        }
        
        // 计算每行之间的间距
        float rowSpacing = patrolHeight / (rowCount - 1);
        
        for (int i = 0; i < rowCount; i++)
        {
            // 计算当前行的Z坐标
            float zPos = patrolStartPoint.z + i * rowSpacing;
            
            // 偶数行从左到右，奇数行从右到左（形成Z形）
            if (i % 2 == 0)
            {
                // 左到右
                trajectoryPoints.Add(new Vector3(patrolStartPoint.x, patrolStartPoint.y, zPos));
                trajectoryPoints.Add(new Vector3(patrolStartPoint.x + patrolWidth, patrolStartPoint.y, zPos));
            }
            else
            {
                // 右到左
                trajectoryPoints.Add(new Vector3(patrolStartPoint.x + patrolWidth, patrolStartPoint.y, zPos));
                trajectoryPoints.Add(new Vector3(patrolStartPoint.x, patrolStartPoint.y, zPos));
            }
        }
        
        Debug.Log($"【中心人员】已生成Z形巡逻轨迹，共 {trajectoryPoints.Count} 个点，{rowCount} 行");
    }

    /// <summary>
    /// 计算从起点到当前位置的总移动距离（用于调试和数据输出）
    /// </summary>
    /// <returns>总移动距离（m）</returns>
    public float CalculateTotalDistance()
    {
        if (trajectoryPoints.Count < 2 || CurrentTargetIndex == 0) return 0;

        float totalDistance = 0;
        int fullCycles = CurrentTargetIndex / trajectoryPoints.Count;
        int currentIndexInCycle = CurrentTargetIndex % trajectoryPoints.Count;

        // 计算完整循环的距离
        float cycleDistance = 0;
        for (int i = 0; i < trajectoryPoints.Count - 1; i++)
        {
            cycleDistance += Vector3.Distance(trajectoryPoints[i], trajectoryPoints[i + 1]);
        }
        // 添加最后一段回到起点的距离
        cycleDistance += Vector3.Distance(trajectoryPoints[^1], trajectoryPoints[0]);
        
        totalDistance += fullCycles * cycleDistance;

        // 计算当前循环中已移动的距离
        for (int i = 0; i < currentIndexInCycle - 1; i++)
        {
            totalDistance += Vector3.Distance(trajectoryPoints[i], trajectoryPoints[i + 1]);
        }

        // 计算当前轨迹段的已移动距离
        if (currentIndexInCycle - 1 >= 0)
        {
            totalDistance += Vector3.Distance(trajectoryPoints[currentIndexInCycle - 1], CurrentPosition);
        }

        return Mathf.Round(totalDistance * 100) / 100; // 保留2位小数
    }

    /// <summary>
    /// 追加多段轨迹
    /// </summary>
    /// <param name="newPoints">新增的轨迹点（需按顺序排列）</param>
    public void AppendTrajectoryPoints(List<Vector3> newPoints)
    {
        if (newPoints == null || newPoints.Count == 0)
        {
            Debug.LogWarning("【中心人员】追加轨迹点为空，忽略操作");
            return;
        }

        // 避免重复添加同一终点
        if (Vector3.Distance(trajectoryPoints[^1], newPoints[0]) <= _arriveThreshold)
        {
            trajectoryPoints.RemoveAt(trajectoryPoints.Count - 1);
        }

        trajectoryPoints.AddRange(newPoints);
        Debug.Log($"【中心人员】追加轨迹点完成！总轨迹点数量：{trajectoryPoints.Count}");
    }
    
    private void OnDrawGizmos()
    {
        UnityEditor.Handles.color = PathColor;
        UnityEditor.Handles.Label(transform.position, "Person");
        if (StaticState.ShowRadius)
            UnityEditor.Handles.DrawWireDisc(transform.position, Vector3.up, scanRadius);

        // 可视化轨迹
        if (trajectoryPoints == null || trajectoryPoints.Count < 2) return;

        Gizmos.color = PathColor;
        // 绘制轨迹线段
        for (int i = 0; i < trajectoryPoints.Count - 1; i++)
        {
            Gizmos.DrawLine(trajectoryPoints[i], trajectoryPoints[i + 1]);
        }
        
        // 绘制最后一点到第一点的线段，表示循环
        Gizmos.DrawLine(trajectoryPoints[^1], trajectoryPoints[0]);
        
        // 绘制轨迹点
        Gizmos.color = Color.red;
        foreach (var point in trajectoryPoints)
        {
            Gizmos.DrawSphere(point, 0.1f);
        }
    }
}
