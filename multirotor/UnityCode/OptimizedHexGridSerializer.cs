using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.IO;
using System;

/// <summary>
/// OptimizedHexGridCompatible 类的数据序列化辅助类
/// 用于将六边形网格数据转换为JSON格式供外部算法使用
/// </summary>
public static class OptimizedHexGridSerializer
{
    /// <summary>
    /// 将 OptimizedHexGridCompatible 中的数据序列化为 JSON 字符串
    /// </summary>
    /// <param name="hexGrid">要序列化的六边形网格实例</param>
    /// <returns>序列化后的 JSON 字符串</returns>
    public static string SerializeToJson(OptimizedHexGrid hexGrid)
    {
        if (hexGrid == null)
        {
            Debug.LogError("OptimizedHexGridCompatible 实例为空，无法序列化");
            return null;
        }

        try
        {
            // 创建要序列化的数据模型
            HexGridDataModel dataModel = new HexGridDataModel
            {
                singleHexRadius = hexGrid.singleHexRadius,
                totalRange = hexGrid.totalRange,
                initialEntropy = hexGrid.initialEntropy,
                minEntropy = hexGrid.minEntropy,
                colorReferenceMax = hexGrid.colorReferenceMax,
                cells = new List<HexCellData>()
            };

            // 获取所有蜂窝单元数据（通过反射获取私有字段）
            var allCellsField = typeof(OptimizedHexGrid).GetField("allCells",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (allCellsField != null)
            {
                if (allCellsField.GetValue(hexGrid) is List<OptimizedHexGrid.HexCell> allCells)
                {
                    foreach (var cell in allCells)
                    {
                        dataModel.cells.Add(new HexCellData
                        {
                            x = Mathf.Round(cell.center.x * 1000f) / 1000f, // 保留3位小数
                            z = Mathf.Round(cell.center.z * 1000f) / 1000f,
                            entropy = Mathf.Round(cell.entropy * 100f) / 100f
                        });
                    }
                }
            }

            // 设置 JSON 序列化器配置
            JsonSerializerSettings settings = new JsonSerializerSettings
            {
                Converters = new List<JsonConverter> { new VectorConverter() },
                Formatting = Formatting.Indented
            };

            // 序列化数据
            string json = JsonConvert.SerializeObject(dataModel, settings);
            return json;
        }
        catch (Exception e)
        {
            Debug.LogError("序列化 OptimizedHexGridCompatible 数据时出错: " + e.Message);
            return null;
        }
    }

    /// <summary>
    /// 将 OptimizedHexGridCompatible 中的数据序列化为 JSON 文件
    /// </summary>
    /// <param name="hexGrid">要序列化的六边形网格实例</param>
    /// <param name="filePath">保存的文件路径</param>
    /// <returns>是否保存成功</returns>
    public static bool SerializeToJsonFile(OptimizedHexGrid hexGrid, string filePath)
    {
        try
        {
            string json = SerializeToJson(hexGrid);
            if (string.IsNullOrEmpty(json))
                return false;

            // 确保目录存在
            string directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // 写入文件
            File.WriteAllText(filePath, json);
            Debug.Log("六边形网格数据已成功保存到: " + filePath);
            return true;
        }
        catch (Exception e)
        {
            Debug.LogError("保存 OptimizedHexGridCompatible 数据到文件时出错: " + e.Message);
            return false;
        }
    }

    /// <summary>
    /// 从 JSON 字符串反序列化数据到 OptimizedHexGridCompatible
    /// </summary>
    /// <param name="json">JSON 字符串</param>
    /// <param name="hexGrid">目标六边形网格实例</param>
    /// <returns>是否反序列化成功</returns>
    public static bool DeserializeFromJson(string json, OptimizedHexGrid hexGrid)
    {
        if (string.IsNullOrEmpty(json) || hexGrid == null)
            return false;

        try
        {
            // 设置 JSON 序列化器配置
            JsonSerializerSettings settings = new JsonSerializerSettings
            {
                Converters = new List<JsonConverter> { new VectorConverter() }
            };

            // 反序列化数据
            HexGridDataModel dataModel = JsonConvert.DeserializeObject<HexGridDataModel>(json, settings);
            if (dataModel == null)
                return false;

            // 更新网格参数
            hexGrid.singleHexRadius = dataModel.singleHexRadius;
            hexGrid.totalRange = dataModel.totalRange;
            hexGrid.initialEntropy = dataModel.initialEntropy;
            hexGrid.minEntropy = dataModel.minEntropy;
            hexGrid.colorReferenceMax = dataModel.colorReferenceMax;

            // 这里可以根据需要更新蜂窝单元数据
            // 注意：由于 allCells 是私有字段，完全替换可能需要额外的逻辑
            // 简单处理：调用 UpdateGridData 重建网格
            hexGrid.Invoke("UpdateGridData", 0f);

            return true;
        }
        catch (Exception e)
        {
            Debug.LogError("反序列化 OptimizedHexGridCompatible 数据时出错: " + e.Message);
            return false;
        }
    }

    /// <summary>
    /// 从 JSON 文件反序列化数据到 OptimizedHexGridCompatible
    /// </summary>
    /// <param name="filePath">JSON 文件路径</param>
    /// <param name="hexGrid">目标六边形网格实例</param>
    /// <returns>是否反序列化成功</returns>
    public static bool DeserializeFromJsonFile(string filePath, OptimizedHexGrid hexGrid)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                Debug.LogError("JSON 文件不存在: " + filePath);
                return false;
            }

            string json = File.ReadAllText(filePath);
            return DeserializeFromJson(json, hexGrid);
        }
        catch (Exception e)
        {
            Debug.LogError("读取 JSON 文件时出错: " + e.Message);
            return false;
        }
    }

    // 数据模型类，用于 JSON 序列化
    [Serializable]
    private class HexGridDataModel
    {
        public float singleHexRadius;
        public float totalRange;
        public float initialEntropy;
        public float minEntropy;
        public float colorReferenceMax;
        public List<HexCellData> cells;
    }

    [Serializable]
    private class HexCellData
    {
        public float x;
        public float z;
        public float entropy;
    }
}