import numpy as np, re, os.path, matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
def get_meshgrid(data, step=.05, border=.5):
    x1_min, x1_max = data[:, 0].min() - border, data[:, 0].max() + border
    x2_min, x2_max = data[:, 1].min() - border, data[:, 1].max() + border
    return np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
def plot_decision_surface(estimator, x_train, y_train, x_test, y_test, title='', metric=accuracy_score):
    estimator.fit(x_train, y_train)
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    x1_values, x2_values = get_meshgrid(x_train)
    x1_ravel, x2_ravel = x1_values.ravel(), x2_values.ravel()
    mesh_predictions_ravel = estimator.predict(np.c_[x1_ravel, x2_ravel])
    mesh_predictions = np.array(mesh_predictions_ravel).reshape(x1_values.shape)
    plt.grid(False)
    plt.pcolormesh(x1_values, x2_values, mesh_predictions)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=100, edgecolors='black')
    plt.xlabel('Признак 1'), plt.ylabel('Признак 2')
    plt.title('Обучающая выборка, {}={:.2f}'.format(metric.__name__, metric(y_train, estimator.predict(x_train))))
    plt.subplot(1, 2, 2)
    x1_values, x2_values = get_meshgrid(data)
    x1_ravel, x2_ravel = x1_values.ravel(), x2_values.ravel()
    mesh_predictions_ravel = estimator.predict(np.c_[x1_ravel, x2_ravel])
    mesh_predictions = np.array(mesh_predictions_ravel).reshape(x1_values.shape)
    plt.grid(False)
    plt.pcolormesh(x1_values, x2_values, mesh_predictions)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, s=100, edgecolors='black')
    plt.title('Тестовая выборка, {}={:.2f}'.format(metric.__name__, metric(y_test, estimator.predict(x_test))))
    plt.xlabel('Признак 1'), plt.ylabel('Признак 2')
    plt.suptitle(title, fontsize=20)
def read_data_number(s):
    return re.findall(r'-?\d*\.?\d+|\d+', s)
fin = 'input.txt'; dt = 'dataset.txt'; res = 'result.txt'; res_dir = 'result'
print(f'Файл {dt} хранит кластеризованную выборку, последний столбец номер кластера\nФайл {fin} хранит параметры '
      'для генерации выборки и параметры для генерации дерева решений')
char = input(f'Считывать данные из файла {dt}? (y/n) ')
ind = 0; rnd = 0; err = 0
file = open(fin, 'r')
lines_src = file.readlines()
file.close()
if(not os.path.exists(res_dir)):
    os.mkdir(res_dir)
file = open(os.path.join(res_dir, res),'w')
if(char.lower() == 'y'):
    file_d = open(dt,'r')
    lines_dt = file_d.readlines()
    file_d.close()
    data = []; target = []
    for i in range(0, len(lines_dt)):
        lines_dt[i] = lines_dt[i].split()
        data.append([float(lines_dt[i][0]), float(lines_dt[i][1])])
        if(len(lines_dt[i]) == len(lines_dt[0])):
            target.append(int(lines_dt[i][2]))
    if(len(target) != len(data)):
        err = 1
    nf = len(data[0])
    max_target = max(target)
    for i in range(len(target), len(lines_dt)):
        target.append(max_target + 1)
    data = np.asarray(data)
    target = np.asarray(target)
elif(char.lower() == 'n'):
    ns = int(lines_src[0].split()[2]); nf = int(lines_src[1].split()[2]); c = int(lines_src[2].split()[2])
    rnd = int(lines_src[3].split()[2]); cl_std = float(lines_src[4].split()[2])
    sh = True if lines_src[5].split()[2].lower() == 'true' else False
    cb = [float(read_data_number(lines_src[6])[0]), float(read_data_number(lines_src[6])[1])]; ind = 7
    data, target, centers = datasets.make_blobs(n_samples=ns, n_features=nf, centers=c, random_state=rnd,
        cluster_std=cl_std, shuffle=sh, center_box=(cb[0], cb[1]), return_centers=True)
    print('Координаты центров кластеров: ')
    file.write('Координаты центров кластеров:\n')
    for i in range(0, c):
        print(f'{i:6})',sep='',end='')
        file.write(f'{i:6})')
        for j in range(0, nf):
            print(f'{centers[i][j]:30}',sep='',end='')
            file.write(f'{centers[i][j]:30}')
        print()
        file.write('\n')
    print('Сгенерированная выборка: ')
    file.write('Сгенерированная выборка:\n')
    for i in range(0, ns):
        print(f'{i:6})',sep='',end='')
        file.write(f'{i:6})')
        for j in range(0, nf):
            print(f'{data[i][j]:30}',sep='',end='')
            file.write(f'{data[i][j]:30}')
        print(f'{target[i]:10}')
        file.write(f'{target[i]:10}\n')
else:
    print('Неопределенный символ! Программа завершила работу')
    input('Нажмите клавишу Enter...')
    exit()
md = int(lines_src[ind].split()[2]) if lines_src[ind].split()[2] != 'None' else None
mln = int(lines_src[ind + 1].split()[2]) if lines_src[ind + 1].split()[2] != 'None' else None
msl = int(lines_src[ind + 2].split()[2]); crit = (lines_src[ind + 3].split()[2])
test_size = float(lines_src[ind + 4].split()[2]) if float(lines_src[ind + 4].split()[2]) < 1 else int(lines_src[ind + 4].split()[2])
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=rnd, shuffle=False)
tr = DecisionTreeClassifier(random_state=rnd, max_depth=md, criterion=crit, max_leaf_nodes=mln, min_samples_leaf= msl, splitter='best')
tr.fit(x_train, y_train)
y_pred = tr.predict(x_train)
print(f'Результат по обучающей выборке:\n{chr(32):27}',sep='',end='')
file.write(f'Результат по обучающей выборке:\n{chr(32):27}')
for i in range(0, nf):
    if(i != nf - 1):
        print(f'{chr(120) + str(i):30}',sep='',end='')
        file.write(f'{chr(120) + str(i):30}')
    else:
        print(f'{chr(120) + str(i):19}', sep='', end='')
        file.write(f'{chr(120) + str(i):19}')
print(f'{chr(121):7}',sep='',end='')
file.write(f'{chr(121):7}')
print('y_train')
file.write('y_train\n')
for i in range(0, len(x_train)):
    print(f'{i:6})', sep='', end='')
    file.write(f'{i:6})')
    for j in range(0, nf):
        print(f'{x_train[i][j]:30}', sep='', end='')
        file.write(f'{x_train[i][j]:30}')
    print(f'{y_train[i]:10}{y_pred[i]:10}')
    file.write(f'{y_train[i]:10}{y_pred[i]:10}\n')
print()
print(classification_report(y_train, y_pred))
file.write('\n')
file.write(classification_report(y_train, y_pred))
y_pred = tr.predict(x_test)
if(err == 1):
    y_test = y_pred
if(nf == 2):
    plt.figure(figsize=(8, 6))
    if(err == 1):
        plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=100, alpha=0.7)
        plt.scatter(x_test[:, 0], x_test[:, 1], color='red', s=100, alpha=0.7)
    else:
        plt.scatter(data[:, 0], data[:, 1], c=target, s=100, alpha=0.7)
    plt.savefig('result\\Figure_1.png')
    plt.xlabel('Признак 1'), plt.ylabel('Признак 2')
    plot_decision_surface(tr, x_train, y_train, x_test, y_test, metric=accuracy_score)
    plt.savefig('result\\Figure_2.png')
plt.figure(figsize=(8, 6))
feat_n = [f'x{i}' for i in range(0, nf)]
plot_tree(tr, feature_names=feat_n, filled=True)
plt.savefig('result\\Figure_3.png')
print(f'Результат по тестовой выборке:\n{chr(32):27}',sep='',end='')
file.write(f'Результат по тестовой выборке:\n{chr(32):27}')
for i in range(0, nf):
    if(i != nf - 1):
        print(f'{chr(120) + str(i):30}',sep='',end='')
        file.write(f'{chr(120) + str(i):30}')
    else:
        print(f'{chr(120) + str(i):19}', sep='', end='')
        file.write(f'{chr(120) + str(i):19}')
print(f'{chr(121):7}',sep='',end='')
file.write(f'{chr(121):7}')
print('y_test')
file.write('y_test\n')
if(err == 1):
    y_test = []
for i in range(0, len(x_test)):
    if(err == 1):
        y_test.append(' ')
    print(f'{i:6})', sep='', end='')
    file.write(f'{i:6})')
    for j in range(0, nf):
        print(f'{x_test[i][j]:30}', sep='', end='')
        file.write(f'{x_test[i][j]:30}')
    print(f'{y_test[i]:10}{y_pred[i]:10}')
    file.write(f'{y_test[i]:10}{y_pred[i]:10}\n')
if(err == 0):
    print()
    print(classification_report(y_test, y_pred))
    file.write('\n')
    file.write(classification_report(y_test, y_pred))
file.close()
plt.show()
input('Нажмите клавишу Enter...')