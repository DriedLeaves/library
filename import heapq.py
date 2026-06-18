import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import random
from collections import deque
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GreedySearch:
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, grid, start, goal, mode='standard', alpha=2.0, max_expansions=50000):
        self.grid = np.array(grid, dtype=np.int8)
        self.start = start
        self.goal = goal
        self.mode = mode
        self.alpha = alpha
        self.max_expansions = max_expansions
        self.height, self.width = self.grid.shape
        self.heuristic = lambda x, y: abs(x - goal[0]) + abs(y - goal[1])


        self.h_matrix = np.zeros((self.height, self.width), dtype=np.float32)
        for i in range(self.height):
            for j in range(self.width):
                self.h_matrix[i, j] = self.heuristic(i, j)
    
    def is_valid(self, x, y):
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x, y] == 0
    
    def search(self):
        open_set = []
        counter = 0
        parent = {}
        g_cost = {self.start: 0}
        start_h = self.h_matrix[self.start[0], self.start[1]]
        start_f = start_h if self.mode == 'standard' else start_h + self.alpha * g_cost[self.start]
        heapq.heappush(open_set, (start_f, counter, self.start))
        closed_set = set()
        expanded_count = 0
        
        while open_set and expanded_count < self.max_expansions:
            _, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            expanded_count += 1
            if current == self.goal:
                path = []
                node = self.goal
                while node != self.start:
                    path.append(node)
                    node = parent[node]
                path.append(self.start)
                path.reverse()
                return path, expanded_count, closed_set, True
            x, y = current
            for dx, dy in self.DIRECTIONS:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if not self.is_valid(nx, ny) or neighbor in closed_set:
                    continue
                new_g = g_cost[current] + 1
                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor] = new_g
                    parent[neighbor] = current
                    h_val = self.h_matrix[nx, ny]
                    if self.mode == 'standard':
                        f_val = h_val
                    else:
                        f_val = h_val + self.alpha * new_g
                    counter += 1
                    heapq.heappush(open_set, (f_val, counter, neighbor))
        return None, expanded_count, closed_set, False


def generate_random_map(width, height, obstacle_prob, start, goal, ensure_connectivity=True):
    while True:
        grid = [[0 for _ in range(width)] for _ in range(height)]
        for i in range(height):
            for j in range(width):
                if (i, j) == start or (i, j) == goal:
                    continue
                if random.random() < obstacle_prob:
                    grid[i][j] = 1
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0
        if not ensure_connectivity:
            break
        visited = [[False]*width for _ in range(height)]
        q = deque([start])
        visited[start[0]][start[1]] = True
        while q:
            x, y = q.popleft()
            if (x, y) == goal:
                break
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < height and 0 <= ny < width and not visited[nx][ny] and grid[nx][ny]==0:
                    visited[nx][ny] = True
                    q.append((nx,ny))
        if visited[goal[0]][goal[1]]:
            break
    return grid


def visualize_comparison(grid, start, goal,
                         path_std, expanded_std, len_std, num_std,
                         path_w, expanded_w, len_w, num_w,
                         alpha=2.0, title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    def draw(ax, path, expanded, title, plen, pnum):
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                color = 'black' if grid[x][y]==1 else 'white'
                rect = Rectangle((y,x),1,1,facecolor=color,edgecolor='lightgray')
                ax.add_patch(rect)
        if expanded:
            for (x,y) in expanded:
                if (x,y) not in (start, goal):
                    rect = Rectangle((y,x),1,1,facecolor='lightblue',alpha=0.5,edgecolor='none')
                    ax.add_patch(rect)
        if path:
            for (x,y) in path:
                if (x,y) not in (start, goal):
                    rect = Rectangle((y,x),1,1,facecolor='lightgreen',edgecolor='darkgreen')
                    ax.add_patch(rect)
            xs = [p[1]+0.5 for p in path]
            ys = [p[0]+0.5 for p in path]
            ax.plot(xs, ys, color='green', linewidth=2, label='路径')
        ax.text(start[1]+0.5, start[0]+0.5, 'S', ha='center', va='center',
                fontsize=12, color='blue', weight='bold')
        ax.text(goal[1]+0.5, goal[0]+0.5, 'G', ha='center', va='center',
                fontsize=12, color='red', weight='bold')
        info_text = f"路径长度: {plen}\n扩展节点数: {pnum}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(0, len(grid[0])); ax.set_ylim(0, len(grid))
        ax.set_aspect('equal'); ax.invert_yaxis()
        ax.set_title(title, fontsize=14)
        ax.legend(handles=[
            Patch(facecolor='lightblue', alpha=0.5, label='搜索节点'),
            Patch(facecolor='lightgreen', label='最终路径'),
            Patch(facecolor='black', label='障碍物')
        ], loc='upper right')
    draw(ax1, path_std, expanded_std, f"标准贪心 (f = h)", len_std, num_std)
    draw(ax2, path_w, expanded_w, f"加权贪心 (f = h + {alpha}·g)", len_w, num_w)
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=16)
    plt.tight_layout()
    plt.show()


def run_experiment(map_size=80, obstacle_prob=0.4, alpha=2.0, num_trials=50):
    start = (1, 1)
    goal = (map_size-2, map_size-2)
    len_std_list = []
    len_w_list = []
    exp_std_list = []
    exp_w_list = []
    for i in range(num_trials):
        print(f"实验 {i+1}/{num_trials}...", end=' ')
        grid = generate_random_map(map_size, map_size, obstacle_prob, start, goal, ensure_connectivity=True)
        gbfs_std = GreedySearch(grid, start, goal, mode='standard', max_expansions=50000)
        path_std, exp_std, _, ok_std = gbfs_std.search()
        if not ok_std:
            print("标准贪心失败，跳过")
            continue
        gbfs_w = GreedySearch(grid, start, goal, mode='weighted', alpha=alpha, max_expansions=50000)
        path_w, exp_w, _, ok_w = gbfs_w.search()
        if not ok_w:
            print("加权贪心失败，跳过")
            continue
        len_std_list.append(len(path_std))
        len_w_list.append(len(path_w))
        exp_std_list.append(exp_std)
        exp_w_list.append(exp_w)
        print(f"标准长度={len(path_std)}, 加权长度={len(path_w)}")
    n = len(len_std_list)
    if n == 0:
        print("没有成功实验！")
        return
    mean_len_std = np.mean(len_std_list)
    std_len_std = np.std(len_std_list)
    mean_len_w = np.mean(len_w_list)
    std_len_w = np.std(len_w_list)
    mean_exp_std = np.mean(exp_std_list)
    std_exp_std = np.std(exp_std_list)
    mean_exp_w = np.mean(exp_w_list)
    std_exp_w = np.std(exp_w_list)
    print("\n" + "="*60)
    print(f"实验次数: {n}")
    print(f"标准贪心: 平均路径长度 = {mean_len_std:.2f} ± {std_len_std:.2f}")
    print(f"加权贪心: 平均路径长度 = {mean_len_w:.2f} ± {std_len_w:.2f}")
    print(f"路径长度平均缩短: {mean_len_std - mean_len_w:.2f} 步 ({(mean_len_std - mean_len_w)/mean_len_std*100:.1f}%)")
    print(f"标准贪心: 平均扩展节点数 = {mean_exp_std:.2f} ± {std_exp_std:.2f}")
    print(f"加权贪心: 平均扩展节点数 = {mean_exp_w:.2f} ± {std_exp_w:.2f}")
    print(f"扩展节点平均增加: {mean_exp_w - mean_exp_std:.2f} 个 ({(mean_exp_w - mean_exp_std)/mean_exp_std*100:.1f}%)")


if __name__ == "__main__":
    MAP_SIZE = 80          
    OBSTACLE_PROB = 0.4    
    ALPHA = 0.5           
    NUM_TRIALS = 50        
    
    random.seed(42)
    start = (1, 1)
    goal = (MAP_SIZE-2, MAP_SIZE-2)
    grid_viz = generate_random_map(MAP_SIZE, MAP_SIZE, OBSTACLE_PROB, start, goal, ensure_connectivity=True)
    
    gbfs_std = GreedySearch(grid_viz, start, goal, mode='standard', max_expansions=50000)
    path_std, exp_std, closed_std, ok_std = gbfs_std.search()
    if not ok_std:
        path_std, exp_std, closed_std = [], exp_std, set()
    
    gbfs_w = GreedySearch(grid_viz, start, goal, mode='weighted', alpha=ALPHA, max_expansions=50000)
    path_w, exp_w, closed_w, ok_w = gbfs_w.search()
    if not ok_w:
        path_w, exp_w, closed_w = [], exp_w, set()
    
    print("单张地图可视化对比 (80x80, 障碍概率0.4, α=0.5)")
    print(f"标准贪心: 路径长度={len(path_std)}, 扩展节点数={exp_std}")
    print(f"加权贪心: 路径长度={len(path_w)}, 扩展节点数={exp_w}")
    visualize_comparison(grid_viz, start, goal,
                         path_std, closed_std, len(path_std), exp_std,
                         path_w, closed_w, len(path_w), exp_w,
                         ALPHA, f"随机地图对比 ({MAP_SIZE}x{MAP_SIZE}, 障碍{int(OBSTACLE_PROB*100)}%)")
    
    print("\n开始50次实验统计...")
    run_experiment(MAP_SIZE, OBSTACLE_PROB, ALPHA, NUM_TRIALS)