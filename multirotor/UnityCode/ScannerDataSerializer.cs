using System;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;

public class ScannerDataSerializer : MonoBehaviour
{
    // 用于序列化的参数类
    [Serializable]
    public class ScannerParams
    {
        public string UavKey = "";
        // 系数设置(Python内部修改)
        public float repulsionCoefficient;
        public float entropyCoefficient;
        public float distanceCoefficient;
        public float leaderRangeCoefficient;
        public float directionRetentionCoefficient;
        public float updateInterval;

        // 基础参数（Python内部处理）
        public float moveSpeed;
        public float rotationSpeed;
        public float scanRadius;

        // 排斥力参数（Python内部处理）
        public float maxRepulsionDistance;
        public float minSafeDistance;

        // 目标选择策略（Python内部处理）
        public bool avoidRevisits;
        public float targetSearchRange;
        public float revisitCooldown;

        // 位置和方向信息（Unity提供，Python处理）
        public Vector3 position;
        public Vector3 forward;

        // 方向向量（Python提供,Unity绘制）
        public Vector3 scoreDir;
        public Vector3 collideDir;
        public Vector3 pathDir;
        public Vector3 leaderRangeDir;
        public Vector3 directionRetentionDir;
        public Vector3 finalMoveDir;

        // Leader信息（Unity提供）
        public Vector3 leaderPosition;
        public float leaderScanRadius;

        // 已访问蜂窝记录（Unity提供）
        public List<Vector3> visitedCells;
    }


    // 从AutoScanner获取参数并序列化
    public static string SerializeScannerData(AutoScanner scanner)
    {
        if (scanner == null)
            return null;

        var paramsData = new ScannerParams
        {
            // 系数设置
            repulsionCoefficient = scanner.repulsionCoefficient,
            entropyCoefficient = scanner.entropyCoefficient,
            distanceCoefficient = scanner.distanceCoefficient,
            leaderRangeCoefficient = scanner.leaderRangeCoefficient,
            directionRetentionCoefficient = scanner.directionRetentionCoefficient,
            updateInterval = scanner.updateInterval,

            // 基础参数
            moveSpeed = scanner.moveSpeed,
            rotationSpeed = scanner.rotationSpeed,
            scanRadius = scanner.scanRadius,

            // 排斥力参数
            maxRepulsionDistance = scanner.maxRepulsionDistance,
            minSafeDistance = scanner.minSafeDistance,

            // 目标选择策略
            avoidRevisits = scanner.avoidRevisits,
            targetSearchRange = scanner.targetSearchRange,
            revisitCooldown = scanner.revisitCooldown,

            // 位置和方向信息
            position = scanner.transform.position,
            forward = scanner.transform.forward,

            // Leader信息
            leaderPosition = scanner.leader ? scanner.leader.CurrentPosition : Vector3.zero,
            leaderScanRadius = scanner.leader ? scanner.leader.scanRadius : 0f,

            //向量信息 
            scoreDir = scanner.scoreDir,
            collideDir = scanner.collideDir,
            pathDir = scanner.pathDir,
            leaderRangeDir = scanner.leaderRangeDir,
            directionRetentionDir = scanner.directionRetentionDir,
            finalMoveDir = scanner.finalMoveDir,

            // 已访问蜂窝记录（简化版）
            visitedCells = new List<Vector3>()
        };

        var visitedCellsDict = scanner.VisitedCells;


        // 获取已访问蜂窝记录

        if (visitedCellsDict != null)
        {
            foreach (var cellPos in visitedCellsDict.Keys)
            {
                paramsData.visitedCells.Add(cellPos);
            }
        }

        // 序列化为JSON
        return JsonConvert.SerializeObject(paramsData, new VectorConverter());
    }

    // 从JSON反序列化参数到AutoScanner
    public static void DeserializeToScanner(string jsonData, AutoScanner scanner)
    {
        if (string.IsNullOrEmpty(jsonData) || !scanner)
            return;

        try
        {
            var paramsData = JsonConvert.DeserializeObject<ScannerParams>(jsonData);

            if (paramsData != null)
            {
                // 设置系数
                scanner.repulsionCoefficient = paramsData.repulsionCoefficient;
                scanner.entropyCoefficient = paramsData.entropyCoefficient;
                scanner.distanceCoefficient = paramsData.distanceCoefficient;
                scanner.leaderRangeCoefficient = paramsData.leaderRangeCoefficient;
                scanner.directionRetentionCoefficient = paramsData.directionRetentionCoefficient;
                scanner.updateInterval = paramsData.updateInterval;

                // 设置基础参数
                scanner.moveSpeed = paramsData.moveSpeed;
                scanner.rotationSpeed = paramsData.rotationSpeed;
                scanner.scanRadius = paramsData.scanRadius;

                // 设置排斥力参数
                scanner.maxRepulsionDistance = paramsData.maxRepulsionDistance;
                scanner.minSafeDistance = paramsData.minSafeDistance;

                // 设置目标选择策略
                scanner.avoidRevisits = paramsData.avoidRevisits;
                scanner.targetSearchRange = paramsData.targetSearchRange;
                scanner.revisitCooldown = paramsData.revisitCooldown;

                // 目标向量
                scanner.scoreDir = paramsData.scoreDir;
                scanner.collideDir = paramsData.collideDir;
                scanner.pathDir = paramsData.pathDir;
                scanner.leaderRangeDir = paramsData.leaderRangeDir;
                scanner.directionRetentionDir = paramsData.directionRetentionDir;
                scanner.finalMoveDir = paramsData.finalMoveDir;
            }
        }
        catch (Exception e)
        {
            Debug.LogError("反序列化Scanner数据失败: " + e.Message);
        }
    }

    // 保存参数到文件
    public static void SaveParamsToFile(AutoScanner scanner, string filePath)
    {
        try
        {
            string jsonData = SerializeScannerData(scanner);
            System.IO.File.WriteAllText(filePath, jsonData);
            Debug.Log("参数已保存到: " + filePath);
        }
        catch (Exception e)
        {
            Debug.LogError("保存参数失败: " + e.Message);
        }
    }

    // 从文件加载参数
    public static void LoadParamsFromFile(string filePath, AutoScanner scanner)
    {
        try
        {
            if (System.IO.File.Exists(filePath))
            {
                string jsonData = System.IO.File.ReadAllText(filePath);
                DeserializeToScanner(jsonData, scanner);
                Debug.Log("参数已从: " + filePath + " 加载");
            }
            else
            {
                Debug.LogError("文件不存在: " + filePath);
            }
        }
        catch (Exception e)
        {
            Debug.LogError("加载参数失败: " + e.Message);
        }
    }
}