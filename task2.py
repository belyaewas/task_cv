import cv2
import numpy as np

def thinning_zhang_suen(binary_image):
    """
    Утоньшение бинарного изображения (0 и 255) по алгоритму Чжан-Суэна.
    Возвращает скелет (0 и 255).
    """
    skeleton = binary_image.copy().astype(np.uint8)
    skeleton = skeleton // 255   # теперь 0 и 1
    prev = np.zeros(skeleton.shape, np.uint8)

    while True:
        # Шаг 1
        markers = np.zeros(skeleton.shape, np.uint8)
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i, j] == 1:
                    p2 = skeleton[i-1, j]
                    p3 = skeleton[i-1, j+1]
                    p4 = skeleton[i, j+1]
                    p5 = skeleton[i+1, j+1]
                    p6 = skeleton[i+1, j]
                    p7 = skeleton[i+1, j-1]
                    p8 = skeleton[i, j-1]
                    p9 = skeleton[i-1, j-1]

                    A = 0
                    for (p1, p2_) in [(p2, p3), (p3, p4), (p4, p5), (p5, p6), (p6, p7), (p7, p8), (p8, p9), (p9, p2)]:
                        if p1 == 0 and p2_ == 1:
                            A += 1
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if 2 <= B <= 6 and A == 1 and (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                        markers[i, j] = 1
        skeleton[markers == 1] = 0

        # Шаг 2
        markers.fill(0)
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i, j] == 1:
                    p2 = skeleton[i-1, j]
                    p3 = skeleton[i-1, j+1]
                    p4 = skeleton[i, j+1]
                    p5 = skeleton[i+1, j+1]
                    p6 = skeleton[i+1, j]
                    p7 = skeleton[i+1, j-1]
                    p8 = skeleton[i, j-1]
                    p9 = skeleton[i-1, j-1]

                    A = 0
                    for (p1, p2_) in [(p2, p3), (p3, p4), (p4, p5), (p5, p6), (p6, p7), (p7, p8), (p8, p9), (p9, p2)]:
                        if p1 == 0 and p2_ == 1:
                            A += 1
                    B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    if 2 <= B <= 6 and A == 1 and (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                        markers[i, j] = 1
        skeleton[markers == 1] = 0

        if np.array_equal(skeleton, prev):
            break
        prev = skeleton.copy()

    return (skeleton * 255).astype(np.uint8)

def count_neighbors(img):
    """Подсчёт количества 8-соседей (без центра) для каждого пикселя."""
    img_pad = np.pad(img, 1, mode='constant', constant_values=0)
    neighbors = np.zeros_like(img, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbors += img_pad[1+dy:1+dy+img.shape[0], 1+dx:1+dx+img.shape[1]]
    return neighbors

def prune_skeleton(skeleton, max_tail_length=10):
    """
    Удаляет короткие хвосты (концевые ветки) длиной <= max_tail_length.
    skeleton: бинарное изображение (0 и 255).
    Возвращает очищенный скелет.
    """
    skeleton = skeleton.copy()
    # Работаем с маской 0/1
    skel = (skeleton == 255).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        # Находим все концевые точки (1 сосед)
        neighbor_count = count_neighbors(skel)
        endpoints = (skel == 1) & (neighbor_count == 1)
        # Для каждой концевой точки идём вдоль ветки и считаем длину
        h, w = skel.shape
        visited = np.zeros_like(skel, dtype=bool)
        for y in range(h):
            for x in range(w):
                if endpoints[y, x] and not visited[y, x]:
                    # Идём по ветке до первого разветвления или конца
                    length = 0
                    cy, cx = y, x
                    path = [(cy, cx)]
                    visited[cy, cx] = True
                    # Пока у текущей точки ровно 2 соседа (включая предыдущую)
                    while True:
                        nbrs = []
                        for dy in (-1,0,1):
                            for dx in (-1,0,1):
                                if dy==0 and dx==0: continue
                                ny, nx = cy+dy, cx+dx
                                if 0<=ny<h and 0<=nx<w and skel[ny,nx] and not visited[ny,nx]:
                                    nbrs.append((ny,nx))
                        if len(nbrs) == 1:
                            # продолжаем
                            visited[nbrs[0][0], nbrs[0][1]] = True
                            path.append(nbrs[0])
                            cy, cx = nbrs[0]
                            length += 1
                        else:
                            break
                    # Если длина короткая и путь не ведёт к другому концу?
                    # Но достаточно просто: если length <= max_tail_length, удаляем весь путь
                    if length <= max_tail_length:
                        for py, px in path:
                            skel[py, px] = 0
                        changed = True
    return skel * 255

def find_junctions_by_skeleton(image_path,blur_size=9,
                               close_iterations=2,
                               min_junction_distance=10, prune_length=10):
    """
    Находит перекрёстки дорог через утоньшение бинарной маски дорог (HSV).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

    # 1. Выделение дорог через HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.medianBlur(hsv, blur_size)
    #blurred = cv2.GaussianBlur(hsv, (blur_size, blur_size), 15)
    mask = cv2.inRange(blurred, (100, 20, 60), (255, 120, 100))

    # Морфологическое закрытие для соединения разрывов
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=close_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    # 2. Утоньшение (скелетизация)
    skeleton = thinning_zhang_suen(mask)   # возвращает np.uint8 (0 или 255)
    skeleton = prune_skeleton(skeleton, max_tail_length=prune_length)
    if skeleton.dtype != np.uint8:
        skeleton = skeleton.astype(np.uint8)
    kernel_cross = np.array([[1,1,1],
                             [1,1,1],
                             [1,1,1]], dtype=np.uint8)
    skeleton_bool = (skeleton == 255).astype(np.uint8)
    neighbor_count = count_neighbors(skeleton_bool)
    junction_candidates = np.logical_and(skeleton_bool, neighbor_count >= 3)
    #skeleton_bool = (skeleton == 255)
    #junction_candidates = np.logical_and(skeleton_bool, neighbor_count >= 3)

    # 5. Визуализация
    # Группировка близких узлов
    junction_img = junction_candidates.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(junction_img, connectivity=8)
    junction_centers = []
    for label in range(1, num_labels):
        points = np.column_stack(np.where(labels == label))
        center = np.mean(points, axis=0).astype(int)
        junction_centers.append(center)

    # Объединение слишком близких центров
    if len(junction_centers) > 1:
        clustered = []
        used = [False]*len(junction_centers)
        for i, pt in enumerate(junction_centers):
            if not used[i]:
                cluster = [pt]
                used[i] = True
                for j, pt2 in enumerate(junction_centers):
                    if not used[j] and np.linalg.norm(np.array(pt)-np.array(pt2)) <= min_junction_distance:
                        cluster.append(pt2)
                        used[j] = True
                clustered.append(np.mean(cluster, axis=0).astype(int))
        junction_centers = clustered

    junction_count = len(junction_centers)

    # Визуализация

    print(f"Найдено перекрёстков: {junction_count}")
    return junction_count

def metric(actual, ref):
    if ref == 0.0:
        return 1.0 if actual == 0 else 0.0
    rel_diff = abs(actual - ref) / abs(ref)
    if (rel_diff > 1):
        rel_diff = 1
    return max(0.0, 1.0 - rel_diff)

if __name__ == "__main__":
    permas=[]
    ref_data = [6, 2, 2, 7, 12, 4, 7, 6, 3, 4]
    name= ['1.png', '2.png','3.png','4.png','5.png','6.png','7.png', '8.png','9.png','10.png']
    for i in range(len(name)):
      count = find_junctions_by_skeleton(
          name[i],
          blur_size=11,
          close_iterations=1,
          min_junction_distance=150,
          prune_length=50
      )
      print(count)
      permas.append(count)
    score=0
    for i in range(len(permas)):
      score+= metric(permas[i],ref_data[i])
    print(score/10)
