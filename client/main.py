import copy
import logging
import os
import socket
import sys
import time
from itertools import cycle
from threading import Thread
from time import sleep

import pygame

from client.ai import AI
from config import config
from gui import GUI
from req import *
from resp import *

# logger config
logging.basicConfig(
    # uncomment this will redirect log to file *client.log*
    # filename="client.log",
    format="[%(asctime)s][%(levelname)s] %(message)s",
    filemode="a+",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# record the context of global data
gContext = {
    "playerID": None,
    "characterID": [],
    "gameOverFlag": False,
    "prompt": (
        "Take actions!\n"
        "'s': move in current direction\n"
        "'w': turn up\n"
        "'e': turn up right\n"
        "'d': turn down right\n"
        "'x': turn down\n"
        "'z': turn down left\n"
        "'a': turn up left\n"
        "'u': sneak\n"
        "'i': unsneak\n"
        "'j': master weapon attack\n"
        "'k': slave weapon attack\n"
        "Please complete all actions within one frame! \n"
        "[example]: sdq\n"
    ),
    "steps": ["-", "\\", "|", "/"],
    "gameBeginFlag": False,
}


class Client(object):
    """Client obj that send/recv packet.

    Usage:
        >>> with Client() as client: # create a socket according to config file
        >>>     client.connect()     # connect to remote
        >>> 
    """

    def __init__(self) -> None:
        self.config = config
        self.host = self.config.get("Host")
        self.port = self.config.get("Port")
        assert self.host and self.port, "host and port must be provided"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        if self.socket.connect_ex((self.host, self.port)) == 0:
            logger.info(f"connect to {self.host}:{self.port}")
        else:
            logger.error(f"can not connect to {self.host}:{self.port}")
            exit(-1)
        return

    def send(self, req: PacketReq):
        msg = json.dumps(req, cls=JsonEncoder).encode("utf-8")
        length = len(msg)
        self.socket.sendall(length.to_bytes(8, sys.byteorder) + msg)
        # uncomment this will show req packet
        # logger.info(f"send PacketReq, content: {msg}")
        return

    def recv(self):
        length = int.from_bytes(self.socket.recv(8), sys.byteorder)
        result = b''
        while resp := self.socket.recv(length):
            result += resp
            length -= len(resp)
            if length <= 0:
                break

        # uncomment this will show resp packet
        # logger.info(f"recv PacketResp, content: {result}")
        packet = PacketResp().from_json(result)
        return packet

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.socket.close()
        if traceback:
            print(traceback)
            return False
        return True


def cliGetInitReq():
    """Get init request from user input."""
    masterWeaponType = 2  # input("Make choices!\nmaster weapon type: [select from {1-2}]: ")
    slaveWeaponType = 1  # input("slave weapon type: [select from {1-2}]: ")
    return InitReq(
        MasterWeaponType(int(masterWeaponType)), SlaveWeaponType(int(slaveWeaponType))
    )


def cliGetActionReq(characterID: int, ai: AI, resp) -> []:
    direction2action = {
        0: (ActionType.TurnAround, TurnAroundActionParam(Direction.Above)),
        1: (ActionType.TurnAround, TurnAroundActionParam(Direction.TopRight)),
        2: (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomRight)),
        3: (ActionType.TurnAround, TurnAroundActionParam(Direction.Bottom)),
        4: (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomLeft)),
        5: (ActionType.TurnAround, TurnAroundActionParam(Direction.TopLeft)),
    }
    sneaky2action = {
        0: (ActionType.Sneaky, EmptyActionParam()),
        1: (ActionType.UnSneaky, EmptyActionParam()),
    }

    def d2action(d):
        if d < len(direction2action):
            return ActionReq(characterID, *direction2action[d])

    def s2action(s):
        if s < len(sneaky2action):
            return ActionReq(characterID, *sneaky2action[s])

    direction_move, sneaky_move, direction_master, sneaky_master, direction_slave, sneaky_slave, rank = ai.get_action(
        resp)

    move = [d2action(direction_move), s2action(sneaky_move),
            ActionReq(characterID, ActionType.Move, EmptyActionParam())]
    master = [d2action(direction_master), s2action(sneaky_master),
              ActionReq(characterID, ActionType.MasterWeaponAttack, EmptyActionParam())]
    slave = [d2action(direction_slave), s2action(sneaky_slave),
             ActionReq(characterID, ActionType.SlaveWeaponAttack, EmptyActionParam())]

    # (9,3,6) = (??????+??????/????????????/?????????) * (Move,???????????????,???????????????)*??????
    # rank ??????
    # 0 -> move M S
    # 1 -> move S M
    # 2 -> M move S
    # 3 -> M S move
    # 4 -> S move M
    # 5 -> S M move

    rank2actions = {
        0: move + master + slave,
        1: move + slave + master,
        2: master + move + slave,
        3: master + slave + move,
        4: slave + move + master,
        5: slave + master + move
    }

    actionReqs = get_real_arr(rank2actions[rank])
    # print("[Action]", actionReqs)
    return actionReqs


def get_real_arr(arr):
    """
    ??????????????????????????????arr
    """
    arr_copy = copy.deepcopy(arr)
    arr_copy = list(filter(None, arr_copy))
    while '' in arr_copy:
        arr_copy.remove('')
    return arr_copy


def refreshUI(ui: GUI, packet: PacketResp):
    """Refresh the UI according to the response."""
    data = packet.data
    if packet.type == PacketType.ActionResp:
        ui.playerID = data.playerID
        ui.color = data.color
        ui.characters = data.characters
        ui.score = data.score
        ui.kill = data.kill
        for block in data.map.blocks:
            if len(block.objs):
                # print(f"[{block.x},{block.y}]{block.objs}")
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": block.objs,
                }
            else:
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": [],
                }
    # subprocess.run(["clear"])
    ui.display()


def recvAndRefresh(ui: GUI, client: Client, ai: AI, id):
    """Recv packet and refresh ui."""
    global gContext
    resp = client.recv()
    refreshUI(ui, resp)
    if resp.type == PacketType.ActionResp:
        if len(resp.data.characters) and not gContext["gameBeginFlag"]:
            gContext["characterID"] = resp.data.characters[-1].characterID
            gContext["playerID"] = resp.data.playerID
            gContext["gameBeginFlag"] = True

    # print("[CTX]", gContext)
    frame = 0
    while resp.type != PacketType.GameOver:
        t = time.time()
        if gContext["characterID"] is not None:
            # print("[resp]",resp)
            if action := cliGetActionReq(gContext["characterID"], ai=ai, resp=resp):
                actionPacket = PacketReq(PacketType.ActionReq, action)
                client.send(actionPacket)
        resp = client.recv()
        t1 = time.time()
        if frame % 5 == 0:
            refreshUI(ui, resp)
        frame += 1
        t2 = time.time()
        ai.resp(resp)
        t3 = time.time()
        print("\n[????????????]", int((t1 - t) * 1000), "ms [UI??????]", int((t2 - t1) * 1000), "ms [????????????]",
              int((t3 - t2) * 1000), "ms \n[?????????]", int((t3 - t) * 1000), "ms\n")

    print(f"Game Over!")
    ai.save(id)
    for (idx, score) in enumerate(resp.data.scores):
        if gContext["playerID"] == idx:
            print(f"You've got \33[1m{score} score\33[0m")
        else:
            print(f"The other player has got \33[1m{score} score \33[0m")

    if resp.data.result == ResultType.Win:
        print("\33[1mCongratulations! You win! \33[0m")
    elif resp.data.result == ResultType.Tie:
        print("\33[1mEvenly matched opponent \33[0m")
    elif resp.data.result == ResultType.Lose:
        print(
            "\33[1mThe goddess of victory is not on your side this time, but there is still a chance next time!\33[0m"
        )

    gContext["gameOverFlag"] = True
    print("Press any key to exit......")


def listen_event(ai: AI, id):
    for event in pygame.event.get():
        # ????????????????????????????????????
        if event.type == pygame.QUIT:
            ai.save(id)
            # ??????????????????
            # pygame.quit()
            # ????????????
            sys.exit()


def main(id):
    ui = GUI()
    ai = AI(id)
    with Client() as client:
        client.connect()
        initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
        client.send(initPacket)
        print(gContext["prompt"])
        # IO thread to display UI
        t = Thread(target=recvAndRefresh, args=(ui, client, ai, id))
        t.start()

        for c in cycle(gContext["steps"]):
            if gContext["gameBeginFlag"]:
                break
            print(
                f"\r\033[0;32m{c}\033[0m \33[1mWaiting for the other player to connect...\033[0m",
                flush=True,
                end="",
            )
            sleep(0.1)

        print("\nGame Start")
        # IO thread accepts user input and sends requests
        while not gContext["gameOverFlag"]:
            listen_event(ai, id)

        # gracefully shutdown
        t.join()


if __name__ == "__main__":
    main(os.getenv('ID'))
