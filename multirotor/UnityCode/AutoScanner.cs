using System;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Sirenix.OdinInspector;
using UnityEditor;

//此代码为简单算法，后续会将所有数据通过中间件进行传递。Python来重新实现
[RequireComponent(typeof(CharacterController))]
public class AutoScanner : MonoBehaviour
{
    [Header("系数设置（权重 = 系数 / 系数总和）")]
    [Tooltip("排斥力系数")] [Min(0f)] public float repulsionCoefficient = 2f;

    [Tooltip("熵值优先系数")] [Min(0f)] public float entropyCoefficient = 3f;

    [Tooltip("最短路径系数")] [Min(0f)] public float distanceCoefficient = 2f;

    [Tooltip("保持在Leader范围内的系数")] [Min(0f)] public float leaderRangeCoefficient = 3f;

    [Tooltip("方向保持系数（值越高，越倾向于直线移动，减少转弯）")] [Min(0f)] public float directionRetentionCoefficient = 2f;

    [Tooltip("方向更新频率（秒）")] public float updateInterval = 0.2f;

    [BoxGroup("颜色")] public Color baseColor = Color.green;
    [BoxGroup("颜色")] public Color boundColor = new Color(0, 0, 0, 0.1f);
    [BoxGroup("颜色")] public Color collideColor = Color.red;
    [BoxGroup("颜色")] public Color leaderRangeColor = new Color(0, 0.5f, 1f, 0.3f);
    [BoxGroup("颜色")] public Color directionRetentionColor = new Color(0.8f, 0.4f, 0, 0.8f); // 方向保持向量颜色

    [Header("核心引用")]
    public OptimizedHexGrid hexGrid;
    public LeaderController leader;

    [Header("基础参数")]
    [Min(0.1f)] public float moveSpeed = 2f;

    [Min(30f)] public float rotationSpeed = 120f;
    [Range(1f, 10f)] public float scanRadius = 5f;


    [Header("排斥力参数")]
    [Min(1f)] public float maxRepulsionDistance = 5f;

    [Min(0.5f)] public float minSafeDistance = 2f;


    [Header("目标选择策略")]
    public bool avoidRevisits = true;

    [Range(5f, 50f)] public float targetSearchRange = 20f;
    [Range(10f, 60f)] public float revisitCooldown = 60f;

    [Header("外部Python通信配置")]
    [Tooltip("是否启用外部Python处理")] public bool useExternalPythonProcessing = false;

    [Tooltip("发送数据文件路径")] public string pythonInputFilePath = "Assets/StreamingAssets/python_input.json";
    [Tooltip("接收数据文件路径")] public string pythonOutputFilePath = "Assets/StreamingAssets/python_output.json";
    [Tooltip("Python处理间隔（秒）")] public float pythonProcessInterval = 1f;

    private float lastPythonProcessTime = 0f;

    // 内部状态
    private CharacterController charController;
    private float lastUpdateTime;
    private Vector3 previousMoveDir; // 上一帧的移动方向，用于计算方向保持

    // 方向向量 ，这些向量应该由Python进行计算。给到Unity进行渲染
    [HideInInspector] public Vector3 scoreDir;              // 熵最优向量
    [HideInInspector] public Vector3 collideDir;            // 排斥最优向量
    [HideInInspector] public Vector3 pathDir;               // 最短路径向量
    [HideInInspector] public Vector3 leaderRangeDir;        // 保持在Leader范围内的向量
    [HideInInspector] public Vector3 directionRetentionDir; // 方向保持向量（减少转弯）
    [HideInInspector] public Vector3 finalMoveDir;          // 最终移动向量

    // 权重缓存 权重也在python中进行配置
    [HideInInspector] public float repulsionWeight;
    [HideInInspector] public float entropyWeight;
    [HideInInspector] public float distanceWeight;
    [HideInInspector] public float leaderRangeWeight;
    [HideInInspector] public float directionRetentionWeight; // 方向保持权重

    // 已访问蜂窝记录
    [HideInInspector] public readonly Dictionary<Vector3, float> VisitedCells = new Dictionary<Vector3, float>();

    // 调试信息
    [ShowInInspector, ReadOnly] private string debugInfo = "";


    private void Awake()
    {
        charController = GetComponent<CharacterController>();
        if (charController == null)
        {
            charController = gameObject.AddComponent<CharacterController>();
            Debug.LogWarning("自动添加了CharacterController组件");
        }
    }

    private void Start()
    {
        // 自动查找网格组件
        if (hexGrid == null)
        {
            hexGrid = FindObjectOfType<OptimizedHexGrid>();
            if (hexGrid == null)
            {
                Debug.LogError("未找到OptimizedHexGridCompatible组件！");
                enabled = false;
                return;
            }
        }

        // 自动查找Leader
        if (leader == null)
        {
            leader = FindObjectOfType<LeaderController>();
        }

        // 初始化方向向量
        previousMoveDir = transform.forward;
        scoreDir = transform.forward;
        pathDir = transform.forward;
        collideDir = Vector3.zero;
        leaderRangeDir = Vector3.zero;
        directionRetentionDir = transform.forward;
        finalMoveDir = transform.forward;

        // 计算初始权重
        CalculateWeights();
    }

    private void OnValidate()
    {
        // 确保系数不为负
        repulsionCoefficient = Mathf.Max(0f, repulsionCoefficient);
        entropyCoefficient = Mathf.Max(0f, entropyCoefficient);
        distanceCoefficient = Mathf.Max(0f, distanceCoefficient);
        leaderRangeCoefficient = Mathf.Max(0f, leaderRangeCoefficient);
        directionRetentionCoefficient = Mathf.Max(0f, directionRetentionCoefficient);

        // 确保最小安全距离小于最大排斥距离
        if (minSafeDistance >= maxRepulsionDistance)
        {
            minSafeDistance = maxRepulsionDistance * 0.5f;
        }

        // 计算权重（仅在编辑器中更新）
        CalculateWeights();
    }

    /// <summary>
    /// 计算权重：F = 系数 / 系数总和
    /// </summary>
    private void CalculateWeights()
    {
        var total = repulsionCoefficient + entropyCoefficient + distanceCoefficient +
                    leaderRangeCoefficient + directionRetentionCoefficient;

        // 处理所有系数都为0的特殊情况
        if (total < 0.001f)
        {
            repulsionWeight = 0.2f;
            entropyWeight = 0.2f;
            distanceWeight = 0.2f;
            leaderRangeWeight = 0.2f;
            directionRetentionWeight = 0.2f;
            return;
        }

        // 计算各权重
        repulsionWeight = repulsionCoefficient / total;
        entropyWeight = entropyCoefficient / total;
        distanceWeight = distanceCoefficient / total;
        leaderRangeWeight = leaderRangeCoefficient / total;
        directionRetentionWeight = directionRetentionCoefficient / total;
    }

    private void Update()
    {
        if (hexGrid == null) return;

        // 执行扫描
        hexGrid.ScanAreaDecrease(transform.position, scanRadius);

        // 外部Python处理逻辑
        if (useExternalPythonProcessing && Time.time - lastPythonProcessTime >= pythonProcessInterval)
        {
            lastPythonProcessTime = Time.time;

            // 将当前状态序列化并保存到文件
            string fullInputPath = Application.dataPath + "/" + pythonInputFilePath;
            ScannerDataSerializer.SaveParamsToFile(this, fullInputPath);

            // 检查是否有Python处理后的结果
            string fullOutputPath = Application.dataPath + "/" + pythonOutputFilePath;
            if (System.IO.File.Exists(fullOutputPath))
            {
                // 读取并应用Python处理后的结果
                ScannerDataSerializer.LoadParamsFromFile(fullOutputPath, this);

                // 可选：处理后删除输出文件，避免重复处理
                try
                {
                    System.IO.File.Delete(fullOutputPath);
                }
                catch (Exception e)
                {
                    Debug.LogWarning("无法删除输出文件: " + e.Message);
                }

                // 如果使用外部处理，跳过内部计算
                MoveToDirection();
                UpdateDebugInfo();
                return;
            }
        }

        // 定期更新方向（仅当不使用外部Python处理或外部处理未返回结果时执行）
        if (Time.time - lastUpdateTime >= updateInterval)
        {
            lastUpdateTime = Time.time;

            // 保存当前方向作为下一帧的" previousMoveDir"
            previousMoveDir = finalMoveDir;

            // 重新计算权重
            CalculateWeights();

            // 计算各方向向量
            CalculateScoreDirection();
            CalculatePathDirection();
            CalculateRepulsionDirection();
            CalculateLeaderRangeDirection();
            CalculateDirectionRetentionDirection(); // 新增：计算方向保持向量

            // 合并所有向量
            MergeDirections();

            // 清理过期访问记录
            CleanupVisitedRecords();
        }

        // 执行移动
        MoveToDirection();

        // 更新调试信息
        UpdateDebugInfo();
    }

    [Button]
    public void TestPOut()
    {
        ScannerDataSerializer.SaveParamsToFile(this, pythonOutputFilePath);
    }

    /// <summary>
    /// 计算熵最优方向向量
    /// </summary>
    private void CalculateScoreDirection()
    {
        var currentPos = new Vector3(transform.position.x, 0, transform.position.z);
        var candidateCells = GetValidCandidateCells(currentPos);

        if (candidateCells.Count == 0)
        {
            scoreDir = Vector3.zero;
            return;
        }

        // 归一化熵值范围（0-1）
        var minEntropy = candidateCells.Min(c => c.entropy);
        var maxEntropy = candidateCells.Max(c => c.entropy);
        var entropyRange = maxEntropy - minEntropy;
        var allEntropySame = Mathf.Abs(entropyRange) < 0.01f;

        // 计算每个候选蜂窝的分数
        var scoredCells = candidateCells.Select(cell =>
            {
                var distance = Vector3.Distance(currentPos, cell.center);
                var normalizedDistance = Mathf.Clamp01(1 - (distance / targetSearchRange));

                // 计算熵值分数
                var entropyScore = allEntropySame ? 0.5f : Mathf.InverseLerp(minEntropy, maxEntropy, cell.entropy);

                // 综合分数：熵值为主，距离为辅
                return new
                {
                    Cell = cell,
                    Score = entropyScore * 0.7f + normalizedDistance * 0.3f
                };
            })
            .OrderByDescending(scored => scored.Score)
            .ToList();

        // 选择最高分的蜂窝作为目标
        var bestCell = scoredCells.First().Cell;
        scoreDir = (bestCell.center - currentPos).normalized;

        // 记录访问
        RecordVisitedCell(bestCell.center);
    }

    /// <summary>
    /// 计算最短路径方向向量
    /// </summary>
    private void CalculatePathDirection()
    {
        pathDir = scoreDir;
    }

    /// <summary>
    /// 计算排斥力方向向量
    /// </summary>
    private void CalculateRepulsionDirection()
    {
        collideDir = Vector3.zero;

        // 找到所有其他扫描器
        var otherScanners = FindObjectsOfType<AutoScanner>()
            .Where(s => s != this && s.gameObject.activeInHierarchy)
            .ToArray();

        if (otherScanners.Length == 0)
        {
            return;
        }

        // 计算每台扫描器的排斥力
        foreach (var other in otherScanners)
        {
            var deltaPos = transform.position - other.transform.position;
            var distance = deltaPos.magnitude;

            // 超出排斥范围或距离过近（避免除以零）
            if (distance > maxRepulsionDistance || distance < 0.1f)
                continue;

            // 计算排斥力比例（距离越近，排斥越强）
            var repulsionRatio = CalculateRepulsionRatio(distance);
            collideDir += deltaPos.normalized * repulsionRatio;
        }

        // 归一化排斥方向
        if (collideDir.magnitude > 0.1f)
        {
            collideDir = collideDir.normalized;
        }
    }

    /// <summary>
    /// 计算保持在Leader范围内的方向向量
    /// </summary>
    private void CalculateLeaderRangeDirection()
    {
        leaderRangeDir = Vector3.zero;

        if (leader == null) return;

        // 计算与Leader的距离
        var distanceToLeader = Vector3.Distance(transform.position, leader.CurrentPosition);

        // 如果超出Leader的范围，生成指向Leader的方向向量
        if (distanceToLeader > leader.scanRadius)
        {
            // 距离越远，返回的力度越大
            var rangeRatio = Mathf.InverseLerp(leader.scanRadius, leader.scanRadius * 2f, distanceToLeader);
            leaderRangeDir = (leader.CurrentPosition - transform.position).normalized * (1f + rangeRatio);
        }
        // 如果离Leader过近，生成轻微远离Leader的方向向量
        else if (distanceToLeader < leader.scanRadius * 0.3f)
        {
            leaderRangeDir = (transform.position - leader.CurrentPosition).normalized * 0.3f;
        }
    }

    /// <summary>
    /// 计算方向保持向量（减少转弯）
    /// </summary>
    private void CalculateDirectionRetentionDirection()
    {
        // 方向保持向量与上一帧的移动方向一致
        directionRetentionDir = previousMoveDir;
    }

    /// <summary>
    /// 计算排斥力比例（0~1）
    /// </summary>
    private float CalculateRepulsionRatio(float distance)
    {
        if (distance <= minSafeDistance)
            return 1f;
        if (distance >= maxRepulsionDistance)
            return 0f;

        // 非线性衰减，近距离排斥力增长更快
        var t = (distance - minSafeDistance) / (maxRepulsionDistance - minSafeDistance);
        return 1f - (t * t);
    }

    /// <summary>
    /// 合并所有方向向量，计算最终移动方向
    /// </summary>
    private void MergeDirections()
    {
        // 应用权重合并向量，包含方向保持向量
        finalMoveDir =
            scoreDir * entropyWeight +
            pathDir * distanceWeight +
            collideDir * repulsionWeight +
            leaderRangeDir * leaderRangeWeight +
            directionRetentionDir * directionRetentionWeight; // 新增：方向保持权重

        // 归一化最终方向
        if (finalMoveDir.magnitude > 0.1f)
        {
            finalMoveDir = finalMoveDir.normalized;
        }
        else
        {
            // 如果最终方向接近零，保持当前方向
            finalMoveDir = transform.forward;
        }
    }

    /// <summary>
    /// 根据最终方向移动
    /// </summary>
    private void MoveToDirection()
    {
        if (finalMoveDir == Vector3.zero || charController == null)
            return;

        // 平滑转向
        var targetRot = Quaternion.LookRotation(finalMoveDir);
        transform.rotation = Quaternion.RotateTowards(
            transform.rotation, targetRot, rotationSpeed * Time.deltaTime);

        // 移动
        var moveStep = finalMoveDir * (moveSpeed * Time.deltaTime);
        moveStep.y = -0.1f; // 轻微向下的力，确保接地
        charController.Move(moveStep);
    }

    /// <summary>
    /// 获取有效的候选蜂窝
    /// </summary>
    private List<OptimizedHexGrid.HexCell> GetValidCandidateCells(Vector3 currentPos)
    {
        // 通过反射获取所有蜂窝
        var cellField = typeof(OptimizedHexGrid).GetField(
            "allCells",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
        );

        if (cellField == null)
        {
            Debug.LogError("无法访问网格的蜂窝数据！");
            return new List<OptimizedHexGrid.HexCell>();
        }

        var allCells = cellField.GetValue(hexGrid) as List<OptimizedHexGrid.HexCell>;
        if (allCells == null || allCells.Count == 0)
        {
            return new List<OptimizedHexGrid.HexCell>();
        }

        // 筛选符合条件的蜂窝
        return allCells.Where(cell =>
        {
            // 检查是否在Leader的范围内
            if (leader != null && Vector3.Distance(cell.center, leader.CurrentPosition) > leader.scanRadius)
            {
                return false; // 不在Leader范围内，跳过
            }

            var distanceToCell = Vector3.Distance(currentPos, cell.center);
            var inSearchRange = distanceToCell <= targetSearchRange;

            // 检查是否需要避免重复访问
            if (avoidRevisits)
            {
                var isVisited = VisitedCells.ContainsKey(cell.center);
                if (isVisited && Time.time - VisitedCells[cell.center] < revisitCooldown)
                {
                    return false; // 仍在冷却期，跳过
                }
            }

            return inSearchRange;
        }).ToList();
    }

    /// <summary>
    /// 记录已访问的蜂窝
    /// </summary>
    private void RecordVisitedCell(Vector3 cellCenter)
    {
        if (!avoidRevisits) return;

        // 四舍五入避免浮点数精度问题
        var roundedCenter = new Vector3(
            Mathf.Round(cellCenter.x * 100) / 100,
            0,
            Mathf.Round(cellCenter.z * 100) / 100
        );

        VisitedCells[roundedCenter] = Time.time;
    }

    /// <summary>
    /// 清理过期的访问记录
    /// </summary>
    private void CleanupVisitedRecords()
    {
        if (!avoidRevisits) return;

        var expiredKeys = VisitedCells
            .Where(kvp => Time.time - kvp.Value >= revisitCooldown)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var key in expiredKeys)
        {
            VisitedCells.Remove(key);
        }
    }

    /// <summary>
    /// 更新调试信息
    /// </summary>
    private void UpdateDebugInfo()
    {
        var leaderStatus = leader != null
            ? $"与Leader距离: {Vector3.Distance(transform.position, leader.CurrentPosition):F1}m"
            : "未关联Leader";

        // 显示权重信息
        var weightInfo = $"权重分配:\n" +
                         $"  熵值: {entropyWeight:F2} ({entropyCoefficient})\n" +
                         $"  路径: {distanceWeight:F2} ({distanceCoefficient})\n" +
                         $"  排斥: {repulsionWeight:F2} ({repulsionCoefficient})\n" +
                         $"  Leader范围: {leaderRangeWeight:F2} ({leaderRangeCoefficient})\n" +
                         $"  方向保持: {directionRetentionWeight:F2} ({directionRetentionCoefficient})";

        // 计算当前转弯角度（与上一帧方向的夹角）
        var turnAngle = Vector3.Angle(previousMoveDir, finalMoveDir);
        var turnInfo = $"转弯角度: {turnAngle:F1}°";

        debugInfo = $"{leaderStatus}\n" +
                    $"{weightInfo}\n" +
                    $"{turnInfo}\n" +
                    $"最终方向: ({finalMoveDir.x:F2}, {finalMoveDir.z:F2})\n" +
                    $"已访问蜂窝: {VisitedCells.Count}";
    }

    // Handles绘制（仅Editor生效）
    private void OnDrawGizmos()
    {
#if UNITY_EDITOR
        // 绘制Leader范围（如果有Leader）
        // if (leader != null)
        // {
        //     Handles.color = leaderRangeColor;
        //     Handles.DrawWireDisc(leader.CurrentPosition, Vector3.up, leader.scanRadius);
        //     Handles.Label(leader.CurrentPosition + Vector3.up * (leader.scanRadius + 0.3f),
        //         $"Leader范围: {leader.scanRadius}m",
        //         new GUIStyle { normal = { textColor = leaderRangeColor } });
        // }

        // 绘制扫描范围
        Handles.color = baseColor;
        Handles.DrawWireDisc(transform.position, Vector3.up, scanRadius);
        Handles.Label(transform.position + Vector3.up * (scanRadius + 0.3f),
            $"扫描范围: {scanRadius}m",
            new GUIStyle { normal = { textColor = baseColor } });

        // 绘制排斥力范围
        Handles.color = boundColor;
        Handles.DrawWireDisc(transform.position, Vector3.up, maxRepulsionDistance);

        // 绘制各方向向量
        var dirLength = 2f;

        // 熵值方向（绿色）
        if (scoreDir != Vector3.zero)
        {
            Handles.color = Color.green;
            var scoreEnd = transform.position + scoreDir * dirLength;
            Handles.DrawLine(transform.position, scoreEnd);
            Handles.DrawSolidDisc(scoreEnd, Vector3.up, 0.1f);
            Handles.Label(scoreEnd + Vector3.up * 0.2f, "熵值",
                new GUIStyle { normal = { textColor = Color.green } });
        }

        // 排斥方向（红色）
        if (collideDir != Vector3.zero)
        {
            Handles.color = Color.red;
            var collideEnd = transform.position + collideDir * dirLength;
            Handles.DrawLine(transform.position, collideEnd);
            Handles.DrawSolidDisc(collideEnd, Vector3.up, 0.1f);
            Handles.Label(collideEnd + Vector3.up * 0.2f, "排斥",
                new GUIStyle { normal = { textColor = Color.red } });
        }

        // Leader范围方向（蓝色）
        if (leaderRangeDir != Vector3.zero)
        {
            Handles.color = leaderRangeColor;
            var leaderEnd = transform.position + leaderRangeDir * dirLength;
            Handles.DrawLine(transform.position, leaderEnd);
            Handles.DrawSolidDisc(leaderEnd, Vector3.up, 0.1f);
            Handles.Label(leaderEnd + Vector3.up * 0.2f, "Leader",
                new GUIStyle { normal = { textColor = leaderRangeColor } });
        }

        // 方向保持向量（橙色）
        if (directionRetentionDir != Vector3.zero)
        {
            Handles.color = directionRetentionColor;
            var retentionEnd = transform.position + directionRetentionDir * dirLength;
            Handles.DrawLine(transform.position, retentionEnd);
            Handles.DrawSolidDisc(retentionEnd, Vector3.up, 0.1f);
            Handles.Label(retentionEnd + Vector3.up * 0.2f, "保持方向",
                new GUIStyle { normal = { textColor = directionRetentionColor } });
        }

        // 最终方向（紫色）
        if (finalMoveDir != Vector3.zero)
        {
            Handles.color = Color.magenta;
            var dirEnd = transform.position + finalMoveDir * dirLength;
            Handles.DrawLine(transform.position, dirEnd);
            Handles.DrawSolidDisc(dirEnd, Vector3.up, 0.15f);
            Handles.Label(dirEnd + Vector3.up * 0.2f, "移动方向",
                new GUIStyle { normal = { textColor = Color.magenta } });
        }
#endif
    }
}