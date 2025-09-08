using System;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using System.IO;


/// <summary>
/// 6边形地图网格 ，，网格数据将由Unity打包后给到中间件。
/// </summary>
public class OptimizedHexGrid : MonoBehaviour
{
    public static OptimizedHexGrid Instance;


    [Header("单个蜂窝设置")]
    [Range(0.2f, 10f)] public float singleHexRadius = 1f; // 单个蜂窝的半径（从中心到顶点）

    [Header("整体范围设置")]
    public float totalRange = 20f; // 总范围半径

    [Header("样式设置")]
    [Tooltip("边缘颜色是否随熵值变化（关闭则使用固定颜色）")] public bool edgeColorByEntropy = true;

    public Color fixedEdgeColor = Color.white; // 固定边缘颜色（当edgeColorByEntropy为false时使用）
    public Color centerMarkerColor = Color.red;
    public float centerMarkerRadiusRatio = 0.1f;

    [Header("熵值设置")]
    public float initialEntropy = 100f; // 初始熵值

    public bool ShowEntropyValue; // 是否显示熵值
    [Tooltip("每次减少扫描的熵值变化量（随时间）")] public float entropyDecreasePerScan = 5f;
    [Tooltip("每次增加扫描的熵值变化量（随时间）")] public float entropyAddPerScan = 1f;
    public float minEntropy = 0f;                                                // 最小熵值（固定为0）
    public bool colorByEntropy = true;                                           // 是否按熵值着色
    public Color lowEntropyColor = Color.green;                                  // 低熵值颜色
    public Color highEntropyColor = Color.red;                                   // 高熵值颜色
    [Tooltip("颜色插值参考上限（仅用于可视化，不限制实际熵值）")] public float colorReferenceMax = 200f; // 用于颜色过渡的参考值

    // 几何参数
    private float hexSideDistance;
    private float horizontalDistance;
    private float verticalDistance;
    private float centerMarkerRadius;

    // 存储所有蜂窝单元（包含中心点和熵值）
    private List<HexCell> allCells = new List<HexCell>();

    // 记录当前处于「减少扫描范围」的蜂窝（用于优先级处理：减少优先于增加）
    private HashSet<HexCell> _decreasingCells = new HashSet<HexCell>();
    private HashSet<Edge> uniqueEdges = new HashSet<Edge>();

    // 新增：记录每条边所属的蜂窝（用于计算边缘颜色）
    private Dictionary<Edge, List<HexCell>> edgeToCells = new Dictionary<Edge, List<HexCell>>();

    // 蜂窝单元类，包含中心点和熵值
    public class HexCell
    {
        public Vector3 center;
        public float entropy;

        public HexCell(Vector3 c, float initialEntropy)
        {
            center = c;
            entropy = initialEntropy;
        }
    }

    // 用于存储唯一边缘的结构体
    private struct Edge
    {
        public Vector3 start;
        public Vector3 end;

        public Edge(Vector3 s, Vector3 e) : this()
        {
            if (CompareVectors(s, e) <= 0)
            {
                start = s;
                end = e;
            }
            else
            {
                start = e;
                end = s;
            }
        }

        private int CompareVectors(Vector3 a, Vector3 b)
        {
            return !Mathf.Approximately(a.x, b.x) ? a.x.CompareTo(b.x) : a.z.CompareTo(b.z);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Edge)) return false;
            var edge = (Edge)obj;
            return start.Equals(edge.start) && end.Equals(edge.end);
        }

        public override int GetHashCode()
        {
            var hash = 17;
            hash = hash * 23 + start.x.GetHashCode();
            hash = hash * 23 + start.y.GetHashCode();
            hash = hash * 23 + start.z.GetHashCode();
            hash = hash * 23 + end.x.GetHashCode();
            hash = hash * 23 + end.y.GetHashCode();
            hash = hash * 23 + end.z.GetHashCode();
            return hash;
        }
    }

    private void Awake()
    {
        if (!Instance)
            Instance = this;
    }

    private void OnEnable()
    {
        _decreasingCells = new HashSet<HexCell>();
        CalculateGridParameters();
        UpdateGridData();
    }

    private void OnValidate()
    {
        // 参数合法性校验
        if (initialEntropy < minEntropy) initialEntropy = minEntropy;
        if (entropyDecreasePerScan < 0) entropyDecreasePerScan = 0;
        if (entropyAddPerScan < 0) entropyAddPerScan = 0;
        if (colorReferenceMax <= minEntropy) colorReferenceMax = initialEntropy;

        CalculateGridParameters();
        UpdateGridData();
    }

    // 计算六边形网格的几何参数
    private void CalculateGridParameters()
    {
        if (singleHexRadius <= 0)
        {
            singleHexRadius = 1f;
            Debug.LogWarning("Hex radius was non-positive, defaulting to 1f");
        }

        const float sqrt3 = 1.73205080757f;
        var hexDiameter = singleHexRadius * sqrt3;

        horizontalDistance = hexDiameter;
        verticalDistance = singleHexRadius * 1.5f;
        centerMarkerRadius = singleHexRadius * centerMarkerRadiusRatio;
    }

    // 更新网格数据：计算所有中心点和唯一边缘（新增边缘与蜂窝的映射）
    private void UpdateGridData()
    {
        allCells.Clear();
        uniqueEdges.Clear();
        edgeToCells.Clear(); // 清空边缘-蜂窝映射
        _decreasingCells.Clear();

        var halfRows = Mathf.FloorToInt(totalRange / verticalDistance) + 1;
        var halfCols = Mathf.FloorToInt(totalRange / horizontalDistance) + 1;

        // 计算所有中心点并初始化熵值
        for (var row = -halfRows; row <= halfRows; row++)
        {
            for (var col = -halfCols; col <= halfCols; col++)
            {
                var center = CalculateHexCenter(row, col);
                if (IsPointInRange(center))
                {
                    allCells.Add(new HexCell(center, initialEntropy));
                }
            }
        }

        // 计算所有六边形的边缘并存储唯一边缘（同时记录边缘所属蜂窝）
        foreach (var cell in allCells)
        {
            AddHexEdges(cell.center, cell); // 传入当前蜂窝，建立映射关系
        }
    }

    // 计算单个蜂窝的中心
    private Vector3 CalculateHexCenter(int row, int col)
    {
        var x = col * horizontalDistance;
        if (row % 2 != 0)
        {
            x += horizontalDistance / 2f;
        }

        var z = row * verticalDistance;
        return new Vector3(x, 0, z);
    }

    // 检查点是否在范围内
    private bool IsPointInRange(Vector3 point)
    {
        return Mathf.Abs(point.x) <= totalRange + singleHexRadius &&
               Mathf.Abs(point.z) <= totalRange + singleHexRadius;
    }

    // 为单个六边形添加边缘到集合（自动去重），并记录边缘所属蜂窝
    private void AddHexEdges(Vector3 center, HexCell cell)
    {
        var vertices = CalculateHexVertices(center);
        for (var i = 0; i < 6; i++)
        {
            var j = (i + 1) % 6;
            var edge = new Edge(vertices[i], vertices[j]);

            uniqueEdges.Add(edge);

            // 建立边缘到蜂窝的映射（一条边属于两个相邻蜂窝）
            if (!edgeToCells.ContainsKey(edge))
            {
                edgeToCells[edge] = new List<HexCell>();
            }

            edgeToCells[edge].Add(cell);
        }
    }

    // 计算六边形的顶点
    private Vector3[] CalculateHexVertices(Vector3 center)
    {
        var vertices = new Vector3[6];
        for (var i = 0; i < 6; i++)
        {
            var angle = Mathf.PI / 180f * (60f * i + 30f);
            var x = center.x + singleHexRadius * Mathf.Cos(angle);
            var z = center.z + singleHexRadius * Mathf.Sin(angle);
            vertices[i] = new Vector3(x, 0, z);
        }

        return vertices;
    }

    #region 扫描功能：减少熵值（优先级高）

    [Tooltip("减少范围内蜂窝的熵值，若与增加扫描重叠，仅执行减少操作")]
    public void ScanAreaDecrease(Vector3 scanCenter, float scanRadius)
    {
        if (scanRadius < 0)
        {
            Debug.LogWarning("减少扫描半径不能为负，已自动修正为0");
            scanRadius = 0;
        }

        scanCenter = new Vector3(scanCenter.x, 0, scanCenter.z);
        _decreasingCells.Clear();

        foreach (var cell in from cell in allCells
                             let distance = Vector3.Distance(cell.center, scanCenter)
                             where distance <= scanRadius
                             select cell)
        {
            _decreasingCells.Add(cell);
            cell.entropy = Mathf.Max(cell.entropy - entropyDecreasePerScan * Time.deltaTime, minEntropy);
        }

        if (Application.isEditor)
            EditorUtility.SetDirty(this);
    }

    #endregion

    #region 扫描功能：增加熵值（优先级低）

    [Tooltip("仅对「不在减少扫描范围内」的蜂窝增加熵值（无上限）")]
    public void ScanAreaIncrease(Vector3 scanCenter, float scanRadius)
    {
        if (scanRadius < 0)
        {
            Debug.LogWarning("增加扫描半径不能为负，已自动修正为0");
            scanRadius = 0;
        }

        scanCenter = new Vector3(scanCenter.x, 0, scanCenter.z);

        foreach (var cell in allCells)
        {
            if (!_decreasingCells.Contains(cell))
            {
                var distance = Vector3.Distance(cell.center, scanCenter);
                if (distance <= scanRadius)
                {
                    cell.entropy += entropyAddPerScan * Time.deltaTime;
                }
            }
        }

        if (Application.isEditor)
            EditorUtility.SetDirty(this);
    }

    #endregion


    private void OnDrawGizmos()
    {
        // 1. 绘制蜂窝边缘（根据熵值着色）
        foreach (var edge in uniqueEdges)
        {
            if (edgeColorByEntropy && edgeToCells.TryGetValue(edge, out var cells))
            {
                // 计算边缘所属蜂窝的平均熵值（一条边通常属于2个相邻蜂窝）
                var averageEntropy = cells.Average(cell => cell.entropy);
                // 按平均熵值计算颜色（与中心标记逻辑一致）
                var normalizedEntropy = Mathf.InverseLerp(minEntropy, colorReferenceMax, averageEntropy);
                normalizedEntropy = Mathf.Clamp01(normalizedEntropy);
                Handles.color = Color.Lerp(lowEntropyColor, highEntropyColor, normalizedEntropy);
            }
            else
            {
                // 不随熵值变化时使用固定颜色
                Handles.color = fixedEdgeColor;
            }

            Handles.DrawLine(edge.start, edge.end);
        }

        // 2. 绘制蜂窝中心标记与熵值
        foreach (var cell in allCells)
        {
            if (colorByEntropy)
            {
                var normalizedEntropy = Mathf.InverseLerp(minEntropy, colorReferenceMax, cell.entropy);
                normalizedEntropy = Mathf.Clamp01(normalizedEntropy);
                Handles.color = Color.Lerp(lowEntropyColor, highEntropyColor, normalizedEntropy);
            }
            else
            {
                Handles.color = centerMarkerColor;
            }

            // 绘制中心圆环
            Handles.DrawWireDisc(cell.center, Vector3.up, centerMarkerRadius);

            ShowEntropyValue = StaticState.ShowAreaValue;
            // 显示熵值
            if (ShowEntropyValue)
            {
                var labelPos = cell.center + Vector3.up * 0.3f;
                Handles.Label(labelPos, cell.entropy.ToString("F0"));
            }
        }
    }

    #region 序列化相关方法

    /// <summary>
    /// 将六边形网格数据序列化为JSON字符串
    /// </summary>
    /// <returns>序列化后的JSON字符串</returns>
    [ContextMenu("Serialize to JSON String")]
    public string SerializeToJson()
    {
        return OptimizedHexGridSerializer.SerializeToJson(this);
    }

    /// <summary>
    /// 将六边形网格数据保存为JSON文件
    /// </summary>
    /// <param name="filePath">保存路径，如果为null则使用默认路径</param>
    /// <returns>是否保存成功</returns>
    [ContextMenu("Save to JSON File")]
    public bool SaveToJsonFile(string filePath = null)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            // 默认保存路径
            filePath = Path.Combine(Application.streamingAssetsPath, "HexGridData.json");
        }

        return OptimizedHexGridSerializer.SerializeToJsonFile(this, filePath);
    }

    /// <summary>
    /// 从JSON字符串加载六边形网格数据
    /// </summary>
    /// <param name="json">JSON字符串</param>
    /// <returns>是否加载成功</returns>
    [ContextMenu("Load from JSON String")]
    public bool LoadFromJson(string json)
    {
        return OptimizedHexGridSerializer.DeserializeFromJson(json, this);
    }

    /// <summary>
    /// 从JSON文件加载六边形网格数据
    /// </summary>
    /// <param name="filePath">文件路径，如果为null则使用默认路径</param>
    /// <returns>是否加载成功</returns>
    [ContextMenu("Load from JSON File")]
    public bool LoadFromJsonFile(string filePath = null)
    {
        if (string.IsNullOrEmpty(filePath))
        {
            // 默认加载路径
            filePath = Path.Combine(Application.streamingAssetsPath, "HexGridData.json");
        }

        return OptimizedHexGridSerializer.DeserializeFromJsonFile(filePath, this);
    }

    #endregion
}