using System;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Sirenix.OdinInspector;
using UnityEditor;

[RequireComponent(typeof(CharacterController))]
public class AutoScanner : MonoBehaviour
{
    [Header("系数设置（权重 = 系数 / 系数总和）")]
    [Tooltip("排斥力系数")]
    [Min(0f)] public float repulsionCoefficient = 2f;
    
    [Tooltip("熵值优先系数")]
    [Min(0f)] public float entropyCoefficient = 3f;
    
    [Tooltip("最短路径系数")]
    [Min(0f)] public float distanceCoefficient = 2f;
    
    [Tooltip("保持在Leader范围内的系数")]
    [Min(0f)] public float leaderRangeCoefficient = 3f;
    
    [Tooltip("方向保持系数（值越高，越倾向于直线移动，减少转弯）")]
    [Min(0f)] public float directionRetentionCoefficient = 2f;
    
    [Tooltip("方向更新频率（秒）")]
    public float updateInterval = 0.2f;


    [BoxGroup("颜色")] 
    public Color baseColor = Color.green;               
    [BoxGroup("颜色")] 
    public Color boundColor = new Color(0, 0, 0, 0.1f); 
    [BoxGroup("颜色")] 
    public Color collideColor = Color.red;              
    [BoxGroup("颜色")] 
    public Color leaderRangeColor = new Color(0, 0.5f, 1f, 0.3f);  
    [BoxGroup("颜色")] 
    public Color directionRetentionColor = new Color(0.8f, 0.4f, 0, 0.8f); // 方向保持向量颜色


    [Header("核心引用")]
    public OptimizedHexGridCompatible hexGrid;
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


    // 内部状态
    private CharacterController charController;
    private float lastUpdateTime;
    private Vector3 previousMoveDir; // 上一帧的移动方向，用于计算方向保持
    
    // 方向向量
    private Vector3 scoreDir;        // 熵最优向量
    private Vector3 collideDir;      // 排斥最优向量
    private Vector3 pathDir;         // 最短路径向量
    private Vector3 leaderRangeDir;  // 保持在Leader范围内的向量
    private Vector3 directionRetentionDir; // 方向保持向量（减少转弯）
    private Vector3 finalMoveDir;    // 最终移动向量
    
    // 权重缓存
    private float repulsionWeight;
    private float entropyWeight;
    private float distanceWeight;
    private float leaderRangeWeight;
    private float directionRetentionWeight; // 方向保持权重
    
    // 已访问蜂窝记录
    private readonly Dictionary<Vector3, float> visitedCells = new Dictionary<Vector3, float>();
    
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
            hexGrid = FindObjectOfType<OptimizedHexGridCompatible>();
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
        float total = repulsionCoefficient + entropyCoefficient + distanceCoefficient + 
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

        // 定期更新方向
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
        
    }

    /// <summary>
    /// 计算熵最优方向向量
    /// </summary>
    private void CalculateScoreDirection()
    {
        Vector3 currentPos = new Vector3(transform.position.x, 0, transform.position.z);
        List<OptimizedHexGridCompatible.HexCell> candidateCells = GetValidCandidateCells(currentPos);

        if (candidateCells.Count == 0)
        {
            scoreDir = Vector3.zero;
            return;
        }

        // 归一化熵值范围（0-1）
        float minEntropy = candidateCells.Min(c => c.entropy);
        float maxEntropy = candidateCells.Max(c => c.entropy);
        float entropyRange = maxEntropy - minEntropy;
        bool allEntropySame = Mathf.Abs(entropyRange) < 0.01f;

        // 计算每个候选蜂窝的分数
        var scoredCells = candidateCells.Select(cell => 
        {
            float distance = Vector3.Distance(currentPos, cell.center);
            float normalizedDistance = Mathf.Clamp01(1 - (distance / targetSearchRange));
            
            // 计算熵值分数
            float entropyScore = allEntropySame ? 0.5f : 
                Mathf.InverseLerp(minEntropy, maxEntropy, cell.entropy);
            
            // 综合分数：熵值为主，距离为辅
            return new { 
                Cell = cell, 
                Score = entropyScore * 0.7f + normalizedDistance * 0.3f 
            };
        })
        .OrderByDescending(scored => scored.Score)
        .ToList();

        // 选择最高分的蜂窝作为目标
        OptimizedHexGridCompatible.HexCell bestCell = scoredCells.First().Cell;
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
        AutoScanner[] otherScanners = FindObjectsOfType<AutoScanner>()
            .Where(s => s != this && s.gameObject.activeInHierarchy)
            .ToArray();

        if (otherScanners.Length == 0)
        {
            return;
        }

        // 计算每台扫描器的排斥力
        foreach (AutoScanner other in otherScanners)
        {
            Vector3 deltaPos = transform.position - other.transform.position;
            float distance = deltaPos.magnitude;

            // 超出排斥范围或距离过近（避免除以零）
            if (distance > maxRepulsionDistance || distance < 0.1f)
                continue;

            // 计算排斥力比例（距离越近，排斥越强）
            float repulsionRatio = CalculateRepulsionRatio(distance);
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
        float distanceToLeader = Vector3.Distance(transform.position, leader.CurrentPosition);
        
        // 如果超出Leader的范围，生成指向Leader的方向向量
        if (distanceToLeader > leader.scanRadius)
        {
            // 距离越远，返回的力度越大
            float rangeRatio = Mathf.InverseLerp(leader.scanRadius, leader.scanRadius * 2f, distanceToLeader);
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
        float t = (distance - minSafeDistance) / (maxRepulsionDistance - minSafeDistance);
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
    /// 获取有效的候选蜂窝
    /// </summary>
    private List<OptimizedHexGridCompatible.HexCell> GetValidCandidateCells(Vector3 currentPos)
    {
        // 通过反射获取所有蜂窝
        var cellField = typeof(OptimizedHexGridCompatible).GetField(
            "allCells", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
        );

        if (cellField == null)
        {
            Debug.LogError("无法访问网格的蜂窝数据！");
            return new List<OptimizedHexGridCompatible.HexCell>();
        }

        var allCells = cellField.GetValue(hexGrid) as List<OptimizedHexGridCompatible.HexCell>;
        if (allCells == null || allCells.Count == 0)
        {
            return new List<OptimizedHexGridCompatible.HexCell>();
        }

        // 筛选符合条件的蜂窝
        return allCells.Where(cell =>
        {
            // 检查是否在Leader的范围内
            if (leader != null && Vector3.Distance(cell.center, leader.CurrentPosition) > leader.scanRadius)
            {
                return false; // 不在Leader范围内，跳过
            }
            
            float distanceToCell = Vector3.Distance(currentPos, cell.center);
            bool inSearchRange = distanceToCell <= targetSearchRange;
            
            // 检查是否需要避免重复访问
            if (avoidRevisits)
            {
                bool isVisited = visitedCells.ContainsKey(cell.center);
                if (isVisited && Time.time - visitedCells[cell.center] < revisitCooldown)
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
        Vector3 roundedCenter = new Vector3(
            Mathf.Round(cellCenter.x * 100) / 100,
            0,
            Mathf.Round(cellCenter.z * 100) / 100
        );

        visitedCells[roundedCenter] = Time.time;
    }

    /// <summary>
    /// 清理过期的访问记录
    /// </summary>
    private void CleanupVisitedRecords()
    {
        if (!avoidRevisits) return;

        List<Vector3> expiredKeys = visitedCells
            .Where(kvp => Time.time - kvp.Value >= revisitCooldown)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (Vector3 key in expiredKeys)
        {
            visitedCells.Remove(key);
        }
    }



}
