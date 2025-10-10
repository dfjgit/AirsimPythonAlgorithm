import pygame
import sys


# 1. 初始化与常量设置
def init_game():
    pygame.init()
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("简易打砖块游戏")

    # 颜色定义
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # 游戏元素参数
    PADDLE_WIDTH = 100
    PADDLE_HEIGHT = 15
    PADDLE_SPEED = 8
    BALL_SIZE = 15
    BALL_SPEED_X = 5
    BALL_SPEED_Y = 5
    BRICK_WIDTH = 70
    BRICK_HEIGHT = 20
    BRICK_GAP = 5
    BRICK_ROWS = 5
    BRICK_COLS = 10
    BRICK_TOP_MARGIN = 50

    return {
        'SCREEN_WIDTH': SCREEN_WIDTH,
        'SCREEN_HEIGHT': SCREEN_HEIGHT,
        'screen': screen,
        'BLACK': BLACK,
        'WHITE': WHITE,
        'RED': RED,
        'GREEN': GREEN,
        'PADDLE_WIDTH': PADDLE_WIDTH,
        'PADDLE_HEIGHT': PADDLE_HEIGHT,
        'PADDLE_SPEED': PADDLE_SPEED,
        'BALL_SIZE': BALL_SIZE,
        'BALL_SPEED_X': BALL_SPEED_X,
        'BALL_SPEED_Y': BALL_SPEED_Y,
        'BRICK_WIDTH': BRICK_WIDTH,
        'BRICK_HEIGHT': BRICK_HEIGHT,
        'BRICK_GAP': BRICK_GAP,
        'BRICK_ROWS': BRICK_ROWS,
        'BRICK_COLS': BRICK_COLS,
        'BRICK_TOP_MARGIN': BRICK_TOP_MARGIN
    }


# 2. 初始化游戏元素
def init_game_elements(config):
    # 挡板（居中底部）
    paddle_x = (config['SCREEN_WIDTH'] - config['PADDLE_WIDTH']) // 2
    paddle_y = config['SCREEN_HEIGHT'] - 40

    # 小球（初始位置在挡板中央上方）
    ball_x = paddle_x + config['PADDLE_WIDTH'] // 2 - config['BALL_SIZE'] // 2
    ball_y = paddle_y - config['BALL_SIZE']
    ball_dx = config['BALL_SPEED_X']  # x方向速度
    ball_dy = -config['BALL_SPEED_Y']  # y方向速度（初始向上）

    # 砖块矩阵（通过二维列表存储砖块位置和状态）
    bricks = []
    for row in range(config['BRICK_ROWS']):
        brick_row = []
        for col in range(config['BRICK_COLS']):
            brick_x = col * (config['BRICK_WIDTH'] + config['BRICK_GAP']) + \
                      (config['SCREEN_WIDTH'] - config['BRICK_COLS'] * (
                                  config['BRICK_WIDTH'] + config['BRICK_GAP'])) // 2
            brick_y = row * (config['BRICK_HEIGHT'] + config['BRICK_GAP']) + config['BRICK_TOP_MARGIN']
            brick_row.append({"rect": pygame.Rect(brick_x, brick_y, config['BRICK_WIDTH'], config['BRICK_HEIGHT']),
                              "active": True})
        bricks.append(brick_row)

    return paddle_x, paddle_y, ball_x, ball_y, ball_dx, ball_dy, bricks


# 3. 检查是否所有砖块都被消除
def check_win_condition(bricks):
    for row in bricks:
        for brick in row:
            if brick["active"]:
                return False  # 还有砖块未被消除
    return True  # 所有砖块都被消除，玩家胜利


# 4. 主游戏函数
def main():
    # 初始化游戏配置
    config = init_game()
    screen = config['screen']

    # 初始化游戏元素
    paddle_x, paddle_y, ball_x, ball_y, ball_dx, ball_dy, bricks = init_game_elements(config)

    # 游戏主循环
    running = True
    clock = pygame.time.Clock()  # 控制游戏帧率

    while running:
        # 控制帧率为60 FPS（避免游戏速度因电脑性能差异过大）
        clock.tick(60)

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 5. 挡板控制
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and paddle_x > 0:
            paddle_x -= config['PADDLE_SPEED']
        if keys[pygame.K_RIGHT] and paddle_x < config['SCREEN_WIDTH'] - config['PADDLE_WIDTH']:
            paddle_x += config['PADDLE_SPEED']

        # 6. 小球移动与碰撞检测
        # 小球位置更新
        ball_x += ball_dx
        ball_y += ball_dy

        # 小球与左右边界碰撞（x方向反弹）
        if ball_x <= 0 or ball_x >= config['SCREEN_WIDTH'] - config['BALL_SIZE']:
            ball_dx = -ball_dx

        # 小球与上边界碰撞（y方向反弹）
        if ball_y <= 0:
            ball_dy = -ball_dy

        # 小球与挡板碰撞（y方向反弹，确保只从上方碰撞）
        paddle_rect = pygame.Rect(paddle_x, paddle_y, config['PADDLE_WIDTH'], config['PADDLE_HEIGHT'])
        ball_rect = pygame.Rect(ball_x, ball_y, config['BALL_SIZE'], config['BALL_SIZE'])

        if ball_rect.colliderect(paddle_rect) and ball_dy > 0:
            ball_dy = -ball_dy

        # 小球与砖块碰撞
        for row in bricks:
            for brick in row:
                if brick["active"] and ball_rect.colliderect(brick["rect"]):
                    brick["active"] = False  # 消除砖块
                    ball_dy = -ball_dy  # 小球反弹
                    break

        # 7. 游戏结束判断（小球落地）
        if ball_y >= config['SCREEN_HEIGHT']:
            font = pygame.font.Font(None, 74)
            text = font.render("Game Over", True, config['RED'])
            screen.blit(text, (200, 250))
            pygame.display.flip()
            pygame.time.wait(3000)  # 等待3秒后退出
            running = False

        # 8. 胜利条件判断（所有砖块被消除）
        if check_win_condition(bricks):
            font = pygame.font.Font(None, 74)
            text = font.render("You Win!", True, config['GREEN'])
            screen.blit(text, (250, 250))
            pygame.display.flip()
            pygame.time.wait(3000)  # 等待3秒后退出
            running = False

        # 9. 绘制所有元素
        screen.fill(config['BLACK'])  # 黑色背景

        # 绘制挡板
        pygame.draw.rect(screen, config['WHITE'], (paddle_x, paddle_y, config['PADDLE_WIDTH'], config['PADDLE_HEIGHT']))

        # 绘制小球
        pygame.draw.circle(screen, config['RED'],
                           (ball_x + config['BALL_SIZE'] // 2, ball_y + config['BALL_SIZE'] // 2),
                           config['BALL_SIZE'] // 2)

        # 绘制砖块
        for row_idx, row in enumerate(bricks):
            for brick in row:
                if brick["active"]:
                    # 不同行砖块用不同颜色
                    color = (0, 255 - row_idx * 40, 255)
                    pygame.draw.rect(screen, color, brick["rect"])

        # 更新屏幕
        pygame.display.flip()

    # 退出游戏
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()