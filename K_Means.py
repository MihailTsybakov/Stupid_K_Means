from matplotlib import pyplot as plt
import pandas as pd
import random, math

def k_means_(X, cluster_count = 2, target = None):
    '''Удаление трагета при надобности'''
    if (target != None):
        X = X.drop(columns = [target])
    ''' Массив меток кластера'''
    clusters = [None for i in range(0, X.shape[0])]
    ''' Центры масс кластеров'''
    centroids = []
    ''' Инициализация центроидов случайными значениями выборки'''
    for i in range(0, cluster_count):
        tmp = [i for i in range(0, X.shape[0])]
        tmp_centroid = []
        rand = random.choice(tmp)
        for i in range(0, X.shape[1]):
            tmp_centroid.append(X.iloc[rand][i])
        centroids.append(tmp_centroid)
    '''Флаг сходимости алгоритма'''
    convergence = False
    '''Проверка сходимости'''
    def check_convergence(centroids_old, centroids_new):
        return (centroids_old == centroids_new)
    old_centroids = []
    while (convergence == False):
        '''Сошёлся ли k means?'''
        if (check_convergence(old_centroids, centroids)):
            convergence = True
            continue
        '''Для каждого обьекта из датафрейма вычисляем расстояния до центроидов и классифицируем'''
        for sample_index in range(0, X.shape[0]):
            sample = X.iloc[sample_index]
            centroids_distance = [math.sqrt(sum(list(map(lambda c,x : (c-x)**2, centroid, sample)))) for centroid in centroids]
            min_dist, min_index = centroids_distance[0], 0
            '''Вычисление ближайшего центроида'''
            for ind, dist in enumerate(centroids_distance):
                if (dist < min_dist):
                    min_dist = dist
                    min_index = ind
            '''Задание соответствующей кластерной метки'''
            clusters[sample_index] = min_index
        '''Пересчёт кластерных центров как центров масс'''
        old_centroids = centroids
        def mean_coord(cluster, coord):
            mean = 0
            for sample in cluster:
                mean += sample[coord]
            return mean/len(cluster)
        for i in range(0, cluster_count):
            tmp_cluster = []
            for obj_index, cluster_mark in enumerate(clusters):
                if (cluster_mark == i):
                    tmp_cluster.append(X.iloc[obj_index])
            if (len(tmp_cluster) == 0):
                '''Если размер кластера оказался 0, положим в него ближайший к центроиду обьект'''
                nearest_sample = X.iloc[0]
                min_dist = math.sqrt(sum(list(map(lambda x,y: (x-y)**2, nearest_sample, centroids[i]))))
                for sample_index in range(1, X.shape[0]):
                    curr_dist = math.sqrt(sum(list(map(lambda x,y: (x-y)**2, centroids[i], X.iloc[sample_index]))))
                    if (curr_dist < min_dist):
                        min_dist = curr_dist
                        nearest_sample = X.iloc[sample_index]
                tmp_cluster.append(nearest_sample)
            center = [mean_coord(tmp_cluster, i) for i in range(0, X.shape[1])]
            centroids[i] = center
    '''Возвращаем метки кластеров'''
    return clusters

''' Оболочка вокруг стандартного k-means для снижения чувствительности к 
    начальной инициализации центроидов'''
def K_Means(X, cluster_count = 2, target = None, optimizing_iterations = 5):
    cluster_marks = None
    def mean_inner_dist(cluster_marks, dataframe):
        mean_dist = 0
        for i in range(0, cluster_count):
           tmp_cluster = []
            for obj_index, cluster_mark in enumerate(cluster_marks):
                if (cluster_mark == i):
                    tmp_cluster.append(dataframe.iloc[obj_index])
            if (len(tmp_cluster) == 0):
                return 10e24
            curr_mean = 0
            for sample in tmp_cluster:
                for sample_ in tmp_cluster:
                    curr_mean += math.sqrt(sum(list(map(lambda x, y: (x-y)**2, sample, sample_))))
            '''
            Метрика - среднее попарное расстояние между обьектами кластера
            '''
            curr_mean /= len(tmp_cluster)
            mean_dist += curr_mean
        return mean_dist/cluster_count
    
    cluster_marks = k_means_(X, cluster_count, target)
    min_inner_dist = mean_inner_dist(cluster_marks, X)
    for i in range(0, optimizing_iterations - 1):
        curr_marks = k_means_(X, cluster_count, target)
        curr_inner_dist = mean_inner_dist(curr_marks, X)
        if (curr_inner_dist < min_inner_dist):
            min_inner_dist = curr_inner_dist
            cluster_marks = curr_marks
    return cluster_marks



'''Сгенерируем случайную выборку для трёх кластеров на плоскости'''
center_1, center_2, center_3 = [1,1], [5,20], [33, 10]
x_1 = [center_1[0] + random.randint(-5,5) for i in range(0,25)]
x_2 = [center_2[0] + random.randint(-7,7) for i in range(0,30)]
x_3 = [center_3[0] + random.randint(-9,9) for i in range(0,28)]

y_1 = [center_1[1] + random.randint(-5,5) for i in range(0,25)]
y_2 = [center_2[1] + random.randint(-7,7) for i in range(0,30)]
y_3 = [center_3[1] + random.randint(-9,9) for i in range(0,28)]

x = x_1 + x_2 + x_3
y = y_1 + y_2 + y_3

# Без кластеризации:
fig, axes = plt.subplots()
axes.scatter(x, y, color = 'grey')

test_df = pd.DataFrame({'x': x, 'y':y})
clusters_marks = K_Means(X = test_df, cluster_count = 3)
x1, x2, x3 = [], [], []
y1, y2, y3 = [], [], []
for index, mark in enumerate(clusters_marks):
    if (mark == 0):
        x1.append(x[index])
        y1.append(y[index])
    elif (mark == 1):
        x2.append(x[index])
        y2.append(y[index])
    elif (mark == 2):
        x3.append(x[index])
        y3.append(y[index])

# С кластеризацией:
fig1, axes1 = plt.subplots()
axes1.scatter(x1, y1, c = 'blue')
axes1.scatter(x2, y2, c = 'red')
axes1.scatter(x3, y3, c = 'green')
