import turtle
import matplotlib.text as txt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from sympy.geometry import point
from scipy import spatial
import candidats
import time

def main():
    start_time = time.clock()
    file_name = 'C:\\Users\\DusaevaYaM\\PycharmProjects\\alg3\\wells_zima_test.txt'
    #file_name = 'C:\\Users\\Yana\\PycharmProjects\\alg3\\wells_zima3.txt'

    cnt = 0
    print('reading------------------------')
    data_list = []
    wells_list = []
    cands_list = []
    current_list = []

    r_hight = 0
    with open(file_name, 'r') as f:
        data = f.readlines()
        for line in data:
            if len(line) < 3:
                continue
            strList = line.split()

            data_list.append([float(strList[1]), float(strList[2])])
            #записываем для всех положений
            if r_hight < math.sqrt(pow(abs(float(strList[5]) - float(strList[3])),2) + pow(abs(float(strList[6]) - float(strList[4])),2)):
                r_hight = math.sqrt(pow(abs(float(strList[5]) - float(strList[3])),2) + pow(abs(float(strList[6]) - float(strList[4])),2))
            wells_list.append((point.Point2D(float(strList[1]), float(strList[2])),
                              point.Point2D(float(strList[1]) - abs(float(strList[5]) - float(strList[3])), float(strList[2]) + abs(float(strList[6]) - float(strList[4])))))
            cand = candidats.cand(strList[0] ,float(strList[1]), float(strList[2]), float(strList[1]) - abs(float(strList[5]) - float(strList[3])), float(strList[2]) + abs(float(strList[6]) - float(strList[4])))
            cands_list.append(cand)
            cnt = cnt + 1
            print(cnt)

    #создаем дерево для поиска
    tree = spatial.KDTree(data_list)
    current_lenght = len(data_list)
    i = -1
    #current_list - все те, которые ни с кем не пересекаются
    #для них можно не менять местоположение
    #wells_list - все конфликтные, для них нужно создать кандидатов
    #кандидаты находятся в cands_list
    #для каждого кандидата задается вероятность правильной расстановки
    num_iter = 0
    print("Prepare")

    #матрица пересечений, где строка - скважина, по столбцам расположены ее кандидаты,
    #а в ячейках - списки пересечений для данного кандидата
    sr_len = 0
    kol_len = 0
    matrix_intersection = np.zeros((len(cands_list), 4))
    for i in range(len(cands_list)):
        cands = cands_list[i]
        flag = True
        j = 2
        #идем до тех пор, пока не найдем первый не пересекающийся
        while flag:
        #for j in range(len(cands_list)):
            flag = False
            p = ([cands.center.x, cands.center.y])
            ls = tree.query(p, k = j)
            #для каждого кандидата
            for k in range(len(cands.cands)):
                c1 = cands.cands[k]
                for c2 in cands_list[ls[1][j-1]].cands:
                #for c2 in cands_list[j].cands:
                    if intersec(c1.point, c2.point):
                        #частные случаи могут быть!!!
                        matrix_intersection[i][k] = matrix_intersection[i][k] + 1
                        c1.intersec.append(c2)
                        #c2.func = c2.func + 1
                        flag = True #идентификатор того, что нашли хотя бы одно пересечение
            sr_len = sr_len + math.sqrt((cands_list[ls[1][j-1]].center.x - cands_list[i].center.x)*(cands_list[ls[1][j-1]].center.x - cands_list[i].center.x)
                                        + (cands_list[ls[1][j-1]].center.y - cands_list[i].center.y)*(cands_list[ls[1][j-1]].center.y - cands_list[i].center.y))
            kol_len = kol_len + 1
            j = j + 1
            if (j == len(cands_list)-1):
                break

    #теперь для всех кандидатов есть их пересечение и количество пересечений
    #теперь нужно убрать все лейблы с максимальным количеством пересечений
    #находим максимальное в матрице и удаляем
    #если в строке все нули - то замечательно, это скважина попадает в current_list со своим первым кандидатом
    num_itre_current = 0
    while True:
        i = 0
        max_intersection = 0
        num_itre_current = num_itre_current + 1
        for num in range(len(cands_list)):
            c1 = cands_list[i]
            max_in_str = max(matrix_intersection[i])
            #могут быть строки, в которых отрицательные числа
            #ищем нулевой
            if (max_in_str == 0):
                for kol in range(4):
                    if matrix_intersection[i][kol] == 0:
                        current_list.append((c1.cands[kol].point, c1.name)) #нужно переделать формат. чтобы можно было записать координаты скважины!!!
                        matrix_intersection = np.delete(matrix_intersection, (i), axis = 0)
                        break
                del cands_list[i]
            else:
                #если есть кандидат без пересечений для данного положения -
                #этот кандидат уходит в current _list
                #для всех остальных кандидатов в этой строке необходимо уменьшить функцию пересечения
                #в строке могут быть отрицательные числа
                min_in_str = max_in_str
                for j in range(4):
                    if (matrix_intersection[i][j] == 0):
                        min_in_str = 0
                if (min_in_str == 0):
                    for j in range(4):
                        #нашли первый нулевой
                        if (matrix_intersection[i][j] == 0):
                            current_list.append((c1.cands[j].point, c1.name))
                            break
                    #для всех кандидатов удаляем пересечение с данными кандидатами
                    for j in range(4):
                        if (matrix_intersection[i][j] != 0):
                            for k in range(len(c1.cands[j].intersec)):
                                cand_intersec = c1.cands[j].intersec[k]
                                for l in range(len(cands_list)):
                                    for m in range(4):
                                        if (cands_list[l].cands[m] == cand_intersec):
                                            matrix_intersection[l][m] = matrix_intersection[l][m] - 1
                    del cands_list[i]
                    matrix_intersection = np.delete(matrix_intersection, (i), axis=0)
                else:
                    i = i + 1
        #если остался только один эелемент, то выходим
        if len(cands_list) == 1:
            break

        #переписали матрицу, теперь в ней нет нулевых элементов
        #нужно найти самый максимальный и удалить его
        for num in range(len(cands_list)):
            c1 = cands_list[num]
            max_in_str = max(matrix_intersection[num])
            if max_intersection < max_in_str:
                max_intersection = max_in_str

        i = 0
        for num in range(len(cands_list)):
            c1 = cands_list[i]
            if (max(matrix_intersection[i]) == max_intersection):
                #если все максимальные, то пропускаем
                """
                kol = 4
                kol_current = 0
                for j in range(4):
                    if matrix_intersection[i][j] == max_intersection:
                        kol_current = kol_current + 1
                    elif (matrix_intersection[i][j] == -1):
                        kol = kol - 1
                if (kol_current == kol):
                    i = i+1
                else:
                """
                for j in range(4):
                    if (matrix_intersection[i][j] == max_intersection):
                        for k in range(len(c1.cands[j].intersec)):
                            cand_intersec = c1.cands[j].intersec[k]
                            for l in range(len(cands_list)):
                                for m in range(4):
                                    if (cands_list[l].cands[m] == cand_intersec):
                                        matrix_intersection[l][m] = matrix_intersection[l][m] - 1
                        matrix_intersection[i][j] = -1
                        break
            else:
                i = i+1

        num_iter = num_iter + 1
        print("num_iter = ", num_iter)
        print("current_lenght = ", len(current_list))
        if len(current_list) == current_lenght:
            break
        if len(current_list) == current_lenght - 1:
            current_list.append((cands_list[0].cands[0].point, cands_list[0].name))
            break
        if num_itre_current == 100:
            break

    #может остаться один элемент ненулевой
    #из него нужно взять кандидата с минимальной функцией пересечения
    if len(cands_list) != 0:
        for i in range(len(cands_list)):
            min_in_str = min(matrix_intersection[i])
            for j in range(4):
                if (matrix_intersection[i][j] == min_in_str):
                    current_list.append((cands_list[i].cands[0].point, cands_list[i].name))
                    break
    print("Success")
    print("Time = ", time.clock() - start_time )
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    for well in current_list:
            x = well[0][0].x
            y = well[0][0].y
            w = (well[0][1].x - well[0][0].x)
            h = (well[0][1].y - well[0][0].y)
            ax1.add_patch(
                patches.Rectangle(
                    (well[0][0].x, well[0][0].y),  # (x,y)
                    (well[0][1].x - well[0][0].x),  # width
                    (well[0][1].y - well[0][0].y),  # height
                    fill=False
                )
            )

    for cntr in current_list:
        plgn = patches.CirclePolygon((cntr[0][0].x, cntr[0][0].y), 1)
        plgn.set_label('txt')
        ax1.add_patch(plgn)
        ax1.text(cntr[0][0].x, cntr[0][0].y, cntr[1], None, False, size=9)
#        txt.Text(cntr._center.x, cntr._center.y, cntr[1])


    ax1.autoscale_view()
    plt.show()

    #для статистики
    #считаем площади пересечения
    intersec_size = []
    lent = 0
    for i in range(len(current_list)):
        j = 2
        flag = True
        #while flag:
        for j in range(len(current_list)):
            flag = False
            p = ([current_list[i][0][0].x, current_list[i][0][0].y])
            #ls = tree.query(p, k=j)
            c1 = current_list[i][0]
            #c2 = current_list[ls[1][j-1]][0]
            c2 = current_list[j][0]
            #если пересекаются - считаем площадь пересечения
            if c1 != c2:
                if intersec(c1, c2):
                    l = max(abs(c1[1].x - c2[0].x), abs(c2[1].x - c1[0].x))
                    h = max(abs(c1[1].y - c2[0].y), abs(c2[1].y - c1[0].y))
                    l = (abs(c1[1].x - c1[0].x) + abs(c2[1].x - c2[0].x)) - l
                    h = (abs(c1[1].y - c1[0].y) + abs(c2[1].y - c2[0].y)) - h
                    intersec_size.append(l*h)
                    flag = True #идентификатор того, что нашли хотя бы одно пересечение
            j = j + 1
            if (j == len(current_list)-1):
                break
    print(intersec_size)
    print("sr len = ", sr_len/kol_len)
    num = -1
    """
    kol = []
    lent = len(intersec_size)
    for i in range(lent):
        num = num + 1
        size = intersec_size[i]
        num2 = -1
        kol.append(1)
        for j in range(i,len(intersec_size) - 1):
            num2 = num2 + 1
            if intersec_size[num2] == size:
                kol[i] = kol[i] + 1
                del intersec_size[num2]
                num2 = num2 - 1
                lent = lent - 1
    print(kol)
    """
    f = open('histogramm.txt', 'w')
    for i in range(len(intersec_size)):
        f.write(str(intersec_size[i]))
        f.write('\n')
    f.close()





def intersec(c1, c2):
    """
    if c1[0].x < c2[0].x:
        if c1[1].x > c2[0].x:
            return True
        else:
            return False
    elif c1[0].x > c2[0].x:
        if c2[1].x < c1[0].x:
            return False
        else:
            return True
    else:
        if (c1[0].y > c2[0].y):
            if (c2[1].y > c1[0].y):
                return True
            else:
                return False
        else:
            if (c1[1].y > c2[0].y):
                return True
            else:
                return False
    """
    w1 = abs(c1[0].x - c1[1].x) / 2
    h1 = abs(c1[0].y - c1[1].y) / 2
    w2 = abs(c2[0].x - c2[1].x) / 2
    h2 = abs(c2[0].y - c2[1].y) / 2
    #находим центр ячеек
    if c1[0].x < c1[1].x:
        center1_x = c1[0].x + w1
    else:
        center1_x = c1[0].x - w1
    if c1[0].y < c1[1].y:
        center1_y = c1[0].y + h1
    else:
        center1_y = c1[0].y - h1

    #находим центр ячеек
    if c2[0].x < c2[1].x:
        center2_x = c2[0].x + w2
    else:
        center2_x = c2[0].x - w2
    if c2[0].y < c2[1].y:
        center2_y = c2[0].y + h2
    else:
        center2_y = c2[0].y - h2


    if ((w1 + w2) >= abs(center1_x - center2_x)) and ((h1 + h2) >= abs(center1_y - center2_y)):
        return True
    else:
        return False






main()



