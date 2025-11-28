const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// 配置路径 - 指向 multirotor 目录下的 data_logs
const DATA_LOGS_DIR = path.join(__dirname, '../multirotor/data_logs');

// 确保目录存在
if (!fs.existsSync(DATA_LOGS_DIR)) {
    fs.mkdirSync(DATA_LOGS_DIR, { recursive: true });
}

// 托管前端页面
app.use(express.static('public'));

// 托管日志文件下载
app.use('/logs', express.static(DATA_LOGS_DIR));

// 实时状态缓存
let currentStatus = {
    connected: false,
    recording: false,
    time: 0,
    unscanned: 0,
    scanned: 0,
    coverage: 0,
    total: 0
};

io.on('connection', (socket) => {
    console.log('新的客户端连接:', socket.id);

    // 发送当前缓存的状态给新连接的前端
    socket.emit('status_update', currentStatus);

    // --- 来自 Python 的事件 ---
    socket.on('py_telemetry', (data) => {
        // 更新缓存
        currentStatus = { ...currentStatus, ...data, connected: true };
        // 广播给所有浏览器
        io.emit('web_telemetry', data);
    });

    socket.on('py_file_saved', (filename) => {
        console.log('文件已保存:', filename);
        io.emit('web_file_saved', filename);
    });

    // --- 来自 浏览器 的事件 ---
    socket.on('cmd_start_record', () => {
        console.log('指令: 开始采集');
        io.emit('py_command', { action: 'start' }); // 转发给 Python
        currentStatus.recording = true;
    });

    socket.on('cmd_stop_record', () => {
        console.log('指令: 停止采集');
        io.emit('py_command', { action: 'stop' }); // 转发给 Python
        currentStatus.recording = false;
    });

    // 获取文件列表接口
    socket.on('get_file_list', () => {
        fs.readdir(DATA_LOGS_DIR, (err, files) => {
            if (err) {
                socket.emit('file_list', []);
                return;
            }
            // 过滤 CSV/JSON 并按时间倒序
            const logFiles = files
                .filter(f => f.endsWith('.csv') || f.endsWith('.json'))
                .map(f => {
                    try {
                        const stat = fs.statSync(path.join(DATA_LOGS_DIR, f));
                        return { name: f, time: stat.mtime };
                    } catch (e) {
                        return { name: f, time: new Date(0) };
                    }
                })
                .sort((a, b) => b.time - a.time);
            socket.emit('file_list', logFiles);
        });
    });

    socket.on('disconnect', () => {
        console.log('客户端断开连接:', socket.id);
    });
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Web控制台运行在: http://localhost:${PORT}`);
    console.log(`数据目录: ${DATA_LOGS_DIR}`);
});

