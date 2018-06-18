import turtle
import matplotlib.text as txt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sympy.geometry import point
from scipy import spatial
import candidats
import time


tree =[]

def main():

    #file_name = 'C:\\Users\\DusaevaYaM\\PycharmProjects\\alg2\\test_light.txt'
    start_time = time.clock()
    file_name = 'C:\\Users\\Yana\\PycharmProjects\\alg2\\wells_zima.txt'

    cnt = 0
    print('reading------------------------')
    data_list = [] #для kd - tree - координаты скважин
    front_list = [] #фронт
    wells_list = [] #все, что слева от фронта
    cands_list = [] #лист с кандидатами

    leftmodel_point = np.zeros(2) #самая левая точка
    with open(file_name, 'r') as f:
        data = f.readlines()
        for line in data:
            if len(line) <  3:
                continue
            strList = line.split()
            if cnt == 0:
                leftmodel_point[0] = float(strList[1])
                leftmodel_point[1] = float(strList[2])
                id = cnt
            else:
                if leftmodel_point[0] > float(strList[1]):
                    leftmodel_point[0] = float(strList[1])
                    leftmodel_point[1] = float(strList[2])
                    id = cnt
            data_list.append([float(strList[1]), float(strList[2])])
            cands_list.append(candidats.cand(strList[0], float(strList[1]), float(strList[2]), abs(float(strList[5]) - float(strList[3])), abs(float(strList[6]) - float(strList[4]))))
            cnt = cnt + 1
            print(cnt)
    statistic_list = data_list.copy()

    #добавляет в фронт самую левую точку
    front_list.append(cands_list[id])
    #одновременно добавляем эту точку в wells_list
    #wells_list.append(cands_list[id])
    del cands_list[id]
    del data_list[id]
    lent = len(cands_list)
    tree = spatial.KDTree(data_list)
    kol = 0
    while True:
        kol = kol + 1
        kol_inner = 0
        print("num iter", kol)
        add = False
        #проходим по всему фронту и находим все те расположения, которые не пересекаются с данным
        #расположение сверху справа
        cand_list = check_front(front_list, cands_list, tree) #кандидаты на добавление в фронт
        #добавляем кандидатов в фронт
        #одновременно удаляем кандидата из поиска
        for candidat in cand_list:
            front_list.append(candidat)
            data_num = -1
            for i in range(len(cands_list)):
                data_num = data_num + 1
                if (cands_list[data_num].center == candidat.center):
                    del data_list[data_num]
                    del cands_list[data_num]
                    data_num = data_num - 1
                    break
        if (len(data_list) > 0):
            tree = spatial.KDTree(data_list)
        # добавляем в wells_list только те скважины, которых там нет
        for front_element in front_list:
            flag = 0
            if (len(wells_list) > 0):
                for wells_element in wells_list:
                    if (front_element.center == wells_element.center):
                        if (len(front_element.cands) == len(wells_element.cands)):
                            flag = 1
                        else:
                            flag = 1
                            del wells_element
                            wells_list.append(front_element)
                if flag == 0:
                    kol_inner = kol_inner + 1
                    wells_list.append(front_element)
            else:
                wells_list.append(front_element)
                kol_inner = kol_inner + 1
        """
        #переписываем фронт
        for p in cand_list:
            for f in front_list:
                for fc in f.cands:
                    for p1 in p.cands:
                        if (fc[0].x >= p1[0].x) and (fc[1].x <= p1[1].x):
                            del fc
                            break
            front_list.append(p)
            k = 0
            for c in cands_list:
                k = k + 1
                if p == c:
                    del cands_list[k-1]
                    del data_list[k-1]
                    if len(data_list) != 0:
                        tree = spatial.KDTree(data_list)
                    break
        """
        print("inner = ", kol_inner)
        if len(wells_list) >= lent + 1:
            break
        """
        #меняем расположение лейбла у ближайших точек, пытаемся найти новую точку для фронта
        (cands, ind_cand) = check_front2(front_list, result_list, tree) #кандидаты на добавление в фронт
        #проверяем всех кандидатов
        # если найдется один, который не пересекается со всем фронтом, то записываем его в фронт
        i = 0
        for p in cands:
            x2 = p[0].x
            y2 = p[0].y
            x22 = p[1].x
            y22 = p[1].y
            flag = False
            for front in front_list:
                x1 = front[0].x
                y1 = front[0].y
                x11 = front[1].x
                y11 = front[1].y
                if intersec(x1, y1, x11, y11, x2, y2, x22, y22):
                    flag = True
            if not flag:
                front_list = make_front(front_list, p)
                wells_list.append(p)
                kol = -1
                for r in result_list:
                    kol = kol + 1
                    if r == p:
                        del r
                        del data_list[kol]
                tree = spatial.KDTree(data_list)
                add = True
            i = i + 1
        #если осталась одна точка, то пока просто добавляем в фронт
        if len(result_list) == 1:
            p = result_list[0]
            front_list = make_front(front_list, p)
            wells_list.append(p)
            break

        if len(result_list) == 0:
            break

        if not add:
            f = front_list[len(front_list) - 1]
            del front_list[len(front_list) - 1]
            x1 = f[0].x
            y1 = f[0].y
            x2 = f[1].x
            y2 = f[1].y
            if y2 > y1:
                y2 = y2 - abs(y1 - y1)*2
            result_list.append((point.Point2D(x1,y1),point.Point2D(x2,y2)))
            data_list.append([x1, y1])
            tree = spatial.KDTree(data_list)
 
        #проходим по всему фронту и находим все те расположения, которые не пересекаются с данным
        #расположение снизу слева
        for p in front_list:
            #текущая точка фронта
            leftmodel_point[0] = p[0][0]
            leftmodel_point[1] = p[0][1]
            i = 2
            #находим для самой левой скважины все пересекающиеся с ней (расположение сверху справа)
            #идем до тех пор, пока не наткнемся на непересекающийся с фронтом
            while True:
                #находим ближайщую точку
                ls = tree.query(leftmodel_point, k=i)
                #поиск ширины, длины начальной точки
                x1 = result_list[ls[1][0]][0].x
                y1 = result_list[ls[1][0]][0].y
                x2 = result_list[ls[1][i - 1]][0].x
                y2 = result_list[ls[1][i - 1]][0].y
                x11 = result_list[ls[1][0]][1].x
                y11 = result_list[ls[1][0]][1].y
                #меняем на расположение снизу слева
                x22 = result_list[ls[1][i - 1]][1].x
                y22 = result_list[ls[1][i - 1]][1].y
                y22 = y22 - (y22 - y2) * 2
                p1 = (point.Point2D(x2, y2), point.Point2D(x22, y22))
                #проверка на пересечение
                if intersec(x1, y1, x11, y11, x2, y2, x22, y22):
                    print("true")
                    i = i + 1
                elif i == len(result_list):
                    break
                else:
                    break
        #если самый ближайщий не пересекался, то перестраиваем фронт
        f =0
        if (i == 2):
            for p in wells_list:
                if (p == p1):
                    f = 1
            if f == 0:
                front_list = make_front(front_list, p1)
                wells_list.append(p1)
                if len(wells_list) == len(result_list):
                    break
                else:
                    continue
        
        #проверяем, пересекается ли последняя точка со всем фронтом
        
        flag = False
        for p in front_list:
            x1 = p[0].x
            y1 = p[0].y
            x11 = p[1].x
            y11 = p[1].y
            x2 = p1[0].x
            y2 = p1[0].y
            x22 = p1[1].x
            y22 = p1[1].y
            if intersec(x1,y1,x11,y11,x2,y2,x22,y22):
                flag = True
        if not flag:
            make_front(front_list,p1)
            wells_list.append(p1)
        """

    for p in wells_list:
        if len(p.cands) > 1:
            for i in range(1, len(p.cands)):
                del p.cands[1]




    #del r_list[0][0]
    #make_data(r_list, left_model)
    print("Success ")
    print("Time = ", time.clock() - start_time)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')

    for rct in wells_list:
        for wel in rct.cands:
            ax1.add_patch(
                patches.Rectangle(
                    (wel[0].x, wel[0].y),  # (x,y)
                    wel[1].x - wel[0].x,  # width
                    wel[1].y - wel[0].y,  # height
                    fill=False
                )
            )

    for cntr in wells_list:
        plgn = patches.CirclePolygon((cntr.center.x, cntr.center.y), 1)
        plgn.set_label('txt')
        ax1.add_patch(plgn)
        ax1.text(cntr.center.x, cntr.center.y, cntr.name, None, False, size=9)
        # txt.Text(cntr._center.x, cntr._center.y, cntr._candidates[0]._id)

    ax1.autoscale_view()
    plt.show()

    #statistic
    tree = spatial.KDTree(statistic_list)
    intersec_list = []
    for well in wells_list:
            leftmodel_point[0] = well.center.x
            leftmodel_point[1] = well.center.y
            candidat = well.cands[0]
            j = 2
            flag = True
            while flag:
                ls = tree.query(leftmodel_point, k=j)
                candidat2 = wells_list[ls[1][j-1]].cands[0]
                x1 = candidat[0].x
                y1 = candidat[0].y
                x11 = candidat[1].x
                y11 = candidat[1].y

                x2 = candidat2[0].x
                y2 = candidat2[0].y
                x22 = candidat2[1].x
                y22 = candidat2[1].y
                if intersec(x1, y1, x11, y11, x2, y2, x22, y22):
                    h1 = abs(candidat[0].x - candidat[1].x)
                    h2 = abs(candidat2[0].x - candidat2[1].x)
                    w1 = abs(candidat[0].y - candidat[1].y)
                    w2 = abs(candidat2[0].y - candidat2[1].y)
                    h_intersec = h1 + h2 - max(abs(candidat2[0].x - candidat[1].x), abs(candidat[0].x - candidat2[1].x))
                    w_intersec = w1 + w2 - max(abs(candidat2[0].y - candidat[1].y), abs(candidat[0].y - candidat2[1].y))
                    intersec_list.append(h_intersec*w_intersec)
                    j = j + 1
                else:
                    flag = False

    print(intersec_list)







def check_front(front_list, result_list, tree):

    leftmodel_point = np.zeros(2)
    cand_list = []
    kol_intersec = []
    sum = 0
    for front_element in front_list:
        # текущая точка фронта
        leftmodel_point[0] = front_element.center.x
        leftmodel_point[1] = front_element.center.y

        for cand1 in front_element.cands:
            j = 2
            # идем до тех пор, пока не наткнемся на непересекающийся с фронтом
            flag = True
            #sum = 0
            while flag:
                # находим ближайщую точку
                ls = tree.query(leftmodel_point, k = j)
                # поиск ширины, длины начальной точки
                for cand2 in result_list[ls[1][j-2]].cands:
                    kol_intersec.append(0)
                    x1 = cand1[0].x
                    y1 = cand1[0].y
                    x11 = cand1[1].x
                    y11 = cand1[1].y

                    x2 = cand2[0].x
                    y2 = cand2[0].y
                    x22 = cand2[1].x
                    y22 = cand2[1].y
                    # проверка на пересечение
                    if intersec(x1, y1, x11, y11, x2, y2, x22, y22):
                        kol_intersec[len(kol_intersec)-1] = kol_intersec[len(kol_intersec)-1] + 1
                        sum = sum + kol_intersec[len(kol_intersec)-1]
                        if len(cand1) > 1:
                            f = 0
                            for c in cand_list:
                                if c == result_list[ls[1][j-2]]:
                                    f = 1
                            if f == 0:
                                cand_list.append(result_list[ls[1][j-2]])
                        else:
                            del cand2
                j = j + 1
                #если не было ни одного пересечения, то просто выходим
                if sum == 0:
                    flag = False
                if j >= (len(result_list) - 1):
                    flag = False
                if len(result_list) == 0:
                    break
    #если не было ни одного пересечения, то добавляем ближайшую точку в фронт
    if sum == 0:
       cand_list.append(result_list[ls[1][j-3]])
    else:
        flag = False
        len_cands_list = len(cand_list)
        for cand_num in range(len_cands_list):
            cands_num = -1
            len_candidats = len(cand_list[cand_num].cands)
            for c_num in range(len_candidats):
                cands_num = cands_num + 1
                c = cand_list[cand_num].cands[cands_num]
                c_list = []
                #находим в фронте все те, которые пересекаются с новым элементом
                for p1 in front_list:
                    for p in p1.cands:
                        x1 = c[0].x
                        y1 = c[0].y
                        x2 = p[0].x
                        y2 = p[0].y
                        x11 = c[1].x
                        y11 = c[1].y
                        x22 = p[1].x
                        y22 = p[1].y
                        if intersec(x1, y1, x11, y11, x2, y2, x22, y22):
                            c_list.append(p)
                for cl in c_list:
                    for i in range(len(front_list)):
                        num_cands = 0
                        for j in range(len(front_list[i].cands)):
                            if cl == front_list[i].cands[num_cands]:
                                if (len(front_list[i].cands) != 1):
                                    del front_list[i].cands[num_cands]
                                else:
                                    flag = True
                            else:
                                num_cands = num_cands + 1
                if flag:
                    if (len(cand_list[cand_num].cands) > 1):
                        del cand_list[cand_num].cands[cands_num]
                        cands_num = cands_num - 1
    return (cand_list)

def make_front(front, p1):
    for f in front:
        for fc in f.cands:
            for p in p1.cands:
                if (fc[0].x >= p[0].x) and (fc[1].x <= p[1].x):
                    del fc
                    break
    front.append(p1)
    return(front)


def intersec(x1, y1, x11, y11, x2, y2, x22, y22):

    w1 = abs(x1 - x11)/2
    h1 = abs(y1 - y11)/2
    w2 = abs(x2 - x22)/2
    h2 = abs(y2 - y22)/2

    #находим центр ячеек
    if x1 < x11:
        center1_x = x1 + w1
    else:
        center1_x = x1 - w1
    if y1 < y11:
        center1_y = y1 + h1
    else:
        center1_y = y1 - h1

    #находим центр ячеек
    if x2 < x22:
        center2_x = x2 + w2
    else:
        center2_x = x2 - w2
    if y2 < y22:
        center2_y = y2 + h2
    else:
        center2_y = y2 - h2

    if ((w1 + w2) >= abs(center1_x - center2_x)) and ((h1 + h2) >= abs(center1_y - center2_y)):
        return True
    else:
        return False


main()
