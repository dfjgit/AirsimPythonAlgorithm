using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

[Serializable]
//数据包数据结构（通信发的这个数据包）
public class DataPacks
{
    [JsonConverter(typeof(StringEnumConverter))]
    public PackType type;  //数据类型

    public string time_span;  //时间戳
    public object pack_data_list;  //通信具体数据
}

[Serializable]
[JsonConverter(typeof(StringEnumConverter))]
//数据包类型
public enum PackType
{
    grid_data,    //HexGridDataModel ，地图数据只解析第一个
    config_data,  //ScannerConfigData
    runtime_data, //ScannerRuntimeData
}