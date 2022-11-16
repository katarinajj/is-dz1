import math
import random

import pygame
import os
import config

from queue import LifoQueue, PriorityQueue, Queue
from itertools import permutations

class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, x, y, file_name, transparent_color=None, wid=config.SPRITE_SIZE, hei=config.SPRITE_SIZE):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (wid, hei))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)


class Surface(BaseSprite):
    def __init__(self):
        super(Surface, self).__init__(0, 0, 'terrain.png', None, config.WIDTH, config.HEIGHT)


class Coin(BaseSprite):
    def __init__(self, x, y, ident):
        self.ident = ident
        super(Coin, self).__init__(x, y, 'coin.png', config.DARK_GREEN)

    def get_ident(self):
        return self.ident

    def position(self):
        return self.rect.x, self.rect.y

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class CollectedCoin(BaseSprite):
    def __init__(self, coin):
        self.ident = coin.ident
        super(CollectedCoin, self).__init__(coin.rect.x, coin.rect.y, 'collected_coin.png', config.DARK_GREEN)

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.RED)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class Agent(BaseSprite):
    def __init__(self, x, y, file_name):
        super(Agent, self).__init__(x, y, file_name, config.DARK_GREEN)
        self.x = self.rect.x
        self.y = self.rect.y
        self.step = None
        self.travelling = False
        self.destinationX = 0
        self.destinationY = 0

    def set_destination(self, x, y):
        self.destinationX = x
        self.destinationY = y
        self.step = [self.destinationX - self.x, self.destinationY - self.y]
        magnitude = math.sqrt(self.step[0] ** 2 + self.step[1] ** 2)
        self.step[0] /= magnitude
        self.step[1] /= magnitude
        self.step[0] *= config.TRAVEL_SPEED
        self.step[1] *= config.TRAVEL_SPEED
        self.travelling = True

    def move_one_step(self):
        if not self.travelling:
            return
        self.x += self.step[0]
        self.y += self.step[1]
        self.rect.x = self.x
        self.rect.y = self.y
        if abs(self.x - self.destinationX) < abs(self.step[0]) and abs(self.y - self.destinationY) < abs(self.step[1]):
            self.rect.x = self.destinationX
            self.rect.y = self.destinationY
            self.x = self.destinationX
            self.y = self.destinationY
            self.travelling = False

    def is_travelling(self):
        return self.travelling

    def place_to(self, position):
        self.x = self.destinationX = self.rect.x = position[0]
        self.y = self.destinationX = self.rect.y = position[1]

    # coin_distance - cost matrix
    # return value - list of coin identifiers (containing 0 as first and last element, as well)
    def get_agent_path(self, coin_distance):
        pass

class ExampleAgent(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        for line in coin_distance:
            print(line)
        path = [i for i in range(1, len(coin_distance))]
        random.shuffle(path)
        return [0] + path + [0]

class Aki(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        coin_cnt = len(coin_distance)
        if coin_cnt < 1:
            return []
        visited = [False] * coin_cnt
        partial_path = [0]
        visited[0] = True
        while len(partial_path) < coin_cnt:
            curr = partial_path[-1]
            next_coin = -1
            for coin in range(0, coin_cnt):
                if visited[coin]:
                    continue

                if next_coin == -1:
                    next_coin = coin
                elif coin_distance[curr][coin] < coin_distance[curr][next_coin]:
                    next_coin = coin

            partial_path.append(next_coin)
            visited[next_coin] = True

        return partial_path + [0]

class Jocke(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        coin_cnt = len(coin_distance)
        if coin_cnt < 1:
            return []

        all_perms = list(permutations(range(1, coin_cnt)))
        # print(all_perms)
        min_cost = math.inf
        path = []
        for perm in all_perms:
            cost = coin_distance[0][perm[0]]
            for i in range(0, len(perm) - 1):
                cost += coin_distance[perm[i]][perm[i + 1]]
            cost += coin_distance[perm[len(perm) - 1]][0]

            if cost < min_cost:
                min_cost = cost
                path = perm

        return [0] + list(path) + [0]

def list_to_str(my_list) -> str:
    my_list = sorted(set(my_list))
    ret = ""
    for x in my_list:
        ret += str(x)
    return ret

def dp_check_skip(dp_expand, curr_partial_path, curr, curr_cost):
    curr_set = curr_partial_path[:-1]  # curr_set = set(curr_partial_path) - {curr}
    key = list_to_str(curr_set) + "-" + str(curr)  # key = "012-3"

    if key not in dp_expand.keys():
        dp_expand[key] = curr_cost
        return False
    if curr_cost >= dp_expand[key]:  # expanding the same partial_path with bigger cost -> skip
        return True

    return False

class Uki(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        partial_path = []
        gen_cnt = 0
        expand_cnt = 0
        pq = PriorityQueue()

        coin_cnt = len(coin_distance)
        if coin_cnt < 1:
            return []
        all_coins = set([coin for coin in range(0, coin_cnt)])

        dp_expand = {}
        partial_path.append([0])
        pq.put((0, -1, 0, gen_cnt))        # (cost, len, coin, gen_cnt)
        gen_cnt += 1
        while not pq.empty():
            curr_cost, len_curr_partial_path, curr, curr_gen_cnt = pq.get()
            # curr_gen_cnt = abs(curr_gen_cnt)
            len_curr_partial_path = abs(len_curr_partial_path)
            curr_partial_path = partial_path[curr_gen_cnt]

            # dp check
            if dp_check_skip(dp_expand, curr_partial_path, curr, curr_cost):
                continue

            expand_cnt += 1
            # print(str(expand_cnt) + ". Expanding " + str(curr) + " with path " + str(curr_partial_path))

            # check for end
            if len_curr_partial_path == coin_cnt + 1:
                return curr_partial_path

            adj = list(all_coins - set(curr_partial_path))
            if len(adj) == 0:
                adj = [0]

            for coin in adj:
                next_cost = curr_cost + coin_distance[curr][coin]
                next_partial_path = curr_partial_path + [coin]
                next_len = len_curr_partial_path + 1

                partial_path.append(next_partial_path)
                pq.put((next_cost, -next_len, coin, gen_cnt))
                gen_cnt += 1

        print("Pozz")
        return [0, 0]

class MSTNode:
    def __init__(self, val, adj_list):
        self.val = val
        self.adj_list = adj_list

def valid_edge(graph, start, end) -> bool:
    # Returns True if from start to end there is NO path in our current graph
    visited = [False] * len(graph)
    queue = Queue()
    queue.put(start)
    while not queue.empty():
        curr = queue.get()
        visited[curr] = True
        for adj in graph[curr].adj_list:
            if adj == end:
                return False

            if not visited[adj]:
                queue.put(adj)

    return True

class DSU:
    def __init__(self, cnt):
        self.par = list(range(0, cnt))

    def get(self, node):
        if self.par[node] == node:
            return node
        self.par[node] = self.get(self.par[node])
        return self.par[node]

    def unite(self, node1, node2):
        # Returns True if node1 and node2 are NOT united, and unites them
        par1 = self.get(node1)
        par2 = self.get(node2)
        if par1 == par2:
            return False
        self.par[par1] = par2
        return True


mst_cache = {"0": 0}

def mst(coins, coin_distance) -> int:
    key = list_to_str(coins)
    if key in mst_cache.keys():
        return mst_cache[key]

    coin_cnt = len(coins)
    if coin_cnt == 0 or coin_cnt == 1:
        return 0

    pq = []
    ret = 0
    edge_cnt = 0
    for row in coins:
        for col in coins:
            if col > row:
                pq.append((coin_distance[row][col], row, col))
    pq.sort(reverse=True)

    dsu = DSU(len(coin_distance))
    graph = [None] * len(coin_distance)
    for coin in coins:
        node = MSTNode(coin, [])
        graph[coin] = node

    while (len(pq) > 0) and (edge_cnt < coin_cnt - 1):
        cost, start, end = pq.pop()

        if valid_edge(graph, start, end):
            graph[start].adj_list.append(end)
            graph[end].adj_list.append(start)
            ret += cost
            edge_cnt += 1
        # if dsu.unite(start, end):
        #     ret += cost
        #     edge_cnt += 1

    # print("RET MST JE " + str(ret))
    mst_cache[key] = ret
    return ret


class Micko(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        partial_path = []
        gen_cnt = 0
        expand_cnt = 0
        pq = PriorityQueue()

        coin_cnt = len(coin_distance)
        all_coins = set([coin for coin in range(0, coin_cnt)])
        if coin_cnt < 1:
            return []

        dp_expand = {}
        partial_path.append([0])
        pq.put((0, -1, 0, gen_cnt, 0))  # (assessment, len, coin, gen_cnt, cost)
        gen_cnt += 1
        while not pq.empty():
            x, len_curr_partial_path, curr, curr_gen_cnt, curr_cost = pq.get()

            len_curr_partial_path = abs(len_curr_partial_path)
            curr_partial_path = partial_path[curr_gen_cnt]

            # dp check
            if dp_check_skip(dp_expand, curr_partial_path, curr, curr_cost):
                continue

            expand_cnt += 1
            #print(str(expand_cnt) + ". Expanding " + str(curr) + " with path " + str(curr_partial_path))

            if len_curr_partial_path == coin_cnt + 1:
                return curr_partial_path

            adj = list(all_coins - set(curr_partial_path))
            if len(adj) == 0:
                adj = [0]

            for coin in adj:
                next_cost = curr_cost + coin_distance[curr][coin]
                next_partial_path = curr_partial_path + [coin]
                next_len = len_curr_partial_path + 1

                coins_for_mst = list(all_coins - set(next_partial_path[1:-1]))
                # print("Coins for mst: " + str(coins_for_mst))

                heuristics = mst(coins_for_mst, coin_distance)

                partial_path.append(next_partial_path)
                pq.put((next_cost + heuristics, -next_len, coin, gen_cnt, next_cost))
                gen_cnt += 1

        print("Pozz")
        return [0, 0]
