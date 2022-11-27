# 导入所需的模块
from typing import List, Union

import pygame

from base import ColorType, ObjType
from resp import Character, Item, SlaveWeapon

SIDE_LENGTH = 10
MIN_LENGTH = SIDE_LENGTH / 2 * 1.732  # 中心与边的距离
CENTER_X = (SIDE_LENGTH + 4) * 0.75 * 31  # MIN_LENGTH * ((15*2)+1) * 1.73 / 2
WINDOW_WIDTH = CENTER_X * 2 + 40  # 窗口宽度
WINDOW_HEIGHT = MIN_LENGTH * 8 + 40  # 窗口高度
TOP_Y = 540  # (0,0) 中心点距离窗口顶部的距离
pygame.init()
f = pygame.font.Font('C:/Windows/Fonts/arial.ttf', 8)

playerID2emoji = {
    0: pygame.image.load("img/cat.png"),
    1: pygame.image.load("img/bee.png"),
    2: pygame.image.load("img/duck.png"),
    3: pygame.image.load("img/bird.png"),
    4: pygame.image.load("img/dead.png"),
}
buff2emoji = {
    1: pygame.image.load("img/hp.png"),
    2: pygame.image.load("img/speed.png")
}
weapon2emoji = {
    1: pygame.image.load("img/kiwi.png"),
    2: pygame.image.load("img/cactus.png")
}


def draw_player(screen, player, block):
    tp1 = pygame.transform.smoothscale(playerID2emoji[player], [20, 20])
    rect = tp1.get_rect()
    rect.center = get_tile_position(block.x, block.y)
    screen.blit(tp1, rect)


def draw_buff(screen, buff, block):
    tp1 = pygame.transform.smoothscale(buff2emoji[buff], [20, 20])
    rect = tp1.get_rect()
    rect.center = get_tile_position(block.x, block.y)
    screen.blit(tp1, rect)


def draw_weapon(screen, wp, block):
    tp1 = pygame.transform.smoothscale(weapon2emoji[wp], [20, 20])
    rect = tp1.get_rect()
    rect.center = get_tile_position(block.x, block.y)
    screen.blit(tp1, rect)


def draw_color(screen, color, block):
    if color == ColorType.White:
        # White
        color = (255, 255, 255)
    elif color == ColorType.Blue:
        color = (64, 103, 255)
    elif color == ColorType.Red:
        color = (255, 77, 98)
    elif color == ColorType.Black:
        color = (0, 0, 0)
    elif color == ColorType.Green:
        color = (107, 225, 141)
    points = get_draw_points(block.x, block.y)
    pygame.draw.polygon(screen, color, points)
    # 以下代码绘制每个色块的坐标
    # str = f"({block.x},{block.y})"
    # txt = f.render(str, True, (255, 0, 255))
    # rect = txt.get_rect()
    # rect.center = get_tile_position(block.x, block.y)
    # screen.blit(txt, rect)


# 每个方块
class Block(object):
    def __init__(
            self,
            x: int,
            y: int,
            color: ColorType = ColorType.White,
            valid: bool = True,
            obj: ObjType = ObjType.Null,
            objData: Union[None, Character, Item, SlaveWeapon] = None,
    ) -> None:
        """The block class used to display.

        Args:
            x (int): x coordinate.
            y (int): y coordinate.
            color (ColorType, optional): Defaults to ColorType.White.
            valid (bool, optional): The block is an obstacle or not.If true, it is not an obstacle, otherwise it is.Defaults to True.
            obj (ObjType, optional): Object on the block. Possible situations are character, item, slaveweapon and no object.Defaults to ObjType.Null.
            objData (_type_, optional): Supplementary information of obj.
                                        If obj is Null, then objData will be ignored.
                                        If obj is type Character, then objdata should be Character instance.
                                        If obj is type Item, then objdata should be Item instance.
                                        If obj is type SlaveWeapon, then objdata should be SlaveWeapon instance.
        """
        self.x = x
        self.y = y
        self.color = color
        self.valid = valid
        self.obj = obj
        self.data = objData

    def draw(self, screen):
        if self.valid:
            if self.obj == ObjType.Null:
                assert isinstance(self.color, ColorType)
                draw_color(screen, self.color, self)
            elif self.obj == ObjType.Character:
                assert isinstance(self.data, Character)
                if not self.data.isAlive:
                    draw_player(screen, 4, self)
                else:
                    draw_player(screen, self.data.playerID, self)
            elif self.obj == ObjType.Item:
                assert isinstance(self.data, Item)
                draw_buff(screen, self.data.buffType, self)
            elif self.obj == ObjType.SlaveWeapon:
                assert isinstance(self.data, SlaveWeapon)
                draw_buff(screen, self.data.weaponType, self)
        else:
            pass
            # 障碍物
            # points = get_draw_points(self.x, self.y)
            # pygame.draw.polygon(screen, (50, 50, 50), points)
            # str = f"({self.x},{self.y})"
            # txt = f.render(str, True, (255, 0, 255))
            # rect = txt.get_rect()
            # rect.center = get_tile_position(self.x, self.y)
            # screen.blit(txt, rect)

    def __str__(self) -> str:
        return f"x:{self.x}, y:{self.y}, color:{self.color}, valid: {self.valid}, obj:{self.obj}, data:{self.data}"


class GUI(object):
    def __init__(
            self,
            playerID: int = 0,
            color: ColorType = ColorType.White,
            characters: List[Character] = [],
            score: int = 0,
            kill: int = 0, ):
        # 使用pygame之前必须初始化
        self.mapSize = 16
        self._blocks = [
            [Block(x, -y) for y in range(self.mapSize)]
            for x in range(self.mapSize)
        ]

        self._playerID = playerID
        self._color = color
        self._characters = characters
        self._score = score
        self._kill = kill
        self._playerID = None
        self._color = None
        # 设置主屏窗口
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
        # 设置窗口的标题，即游戏名称
        pygame.display.set_caption('种子杯')

    def display(self):
        # 引入字体类型
        # 生成文本信息，第一个参数文本内容；第二个参数，字体是否平滑；
        # 第三个参数，RGB模式的字体颜色；第四个参数，RGB模式字体背景颜色；
        # 将准备好的文本信息，绘制到主屏幕 Screen 上。
        # draw_map(self.screen)
        # 固定代码段，实现点击"X"号退出界面的功能，几乎所有的pygame都会使用该段代码
        # # 循环获取事件，监听事件状态

        self.screen.fill((0, 0, 0))
        for x in range(self.mapSize):
            for y in range(self.mapSize):
                self._blocks[x][-y].draw(self.screen)

        if pygame.display.get_active():
            pygame.display.flip()  # 更新屏幕内容

    @property
    def playerID(self):
        return self._playerID

    @playerID.setter
    def playerID(self, playerID):
        if playerID > 0:
            self._playerID = playerID

    @property
    def block(self):
        return self._blocks

    @block.setter
    def block(self, kwargs: dict):
        """Set block attributes.

        supported key value pair:
            {
                "x": int,
                "y": int,
                "color": ColorType,
                "valid": bool,
                "obj": ObjType,
                "objData": data
            }

        """
        block = self._blocks[kwargs.pop("x")][-kwargs.pop("y")]
        for key, value in kwargs.items():
            if hasattr(block, key):
                setattr(block, key, value)

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, characters: List[Character]):
        if (
                isinstance(characters, list)
                and len(characters)
                and all([isinstance(c, Character) for c in characters])
        ):
            self._characters = characters

    @property
    def playerID(self):
        return self._playerID

    @playerID.setter
    def playerID(self, playerID: int):
        if isinstance(playerID, int):
            self._playerID = playerID

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color: ColorType):
        if isinstance(color, ColorType):
            self._color = color

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if score > 0:
            self._score = score

    @property
    def kill(self):
        return self._kill

    @kill.setter
    def kill(self, kill):
        if kill > 0:
            self._kill = kill


def get_tile_position(alpha, beta):
    return - 1.732 * (MIN_LENGTH + 2) * (alpha + beta) + CENTER_X, - (MIN_LENGTH + 2) * (alpha - beta) + TOP_Y


# 获取指定相对坐标的六个角点的绝对坐标用于绘制
# 左上，右上，正右，右下，左下，正左
def get_draw_points(alpha, beta):
    cx, cy = get_tile_position(alpha, beta)
    return (cx - 0.5 * SIDE_LENGTH, cy - MIN_LENGTH), (cx + 0.5 * SIDE_LENGTH, cy - MIN_LENGTH), \
           (cx + SIDE_LENGTH, cy), (cx + 0.5 * SIDE_LENGTH, cy + MIN_LENGTH + 2), \
           (cx - 0.5 * SIDE_LENGTH, cy + MIN_LENGTH + 2), (cx - SIDE_LENGTH, cy)

# if __name__ == '__main__':
#     emoji = f.render(playerID2emoji[0], True,None)
#     rect = emoji.get_rect()
#     rect.center = get_tile_position(0,0)
#     screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_WIDTH))
#     screen.blit(emoji, rect)
#     while True:
#         pygame.display.flip()
