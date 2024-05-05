import os, json, shutil
from tqdm import tqdm
import dataEnhance

res_path = r'D:\selectImg\5-3-no-aug-dataset'
classes = ['peony', 'peony spot',
           'glycyrrhiza uralensis', 'glycyrrhiza uralensis spot',
           'chrysanthemum', 'chrysanthemum spot',
           'salvia', 'salvia spot',
           'glutinosa', 'glutinosa spot',
           'saposhnikovia divaricata', 'saposhnikovia divaricata spot',
           'mint', 'mint spot',
           'aster tataricus', 'aster tataricus spot',
           'sanguisorba', 'sanguisorba spot',
           'insect bite']
pathList = [(r'D:\selectImg\芍药', 0),
            (r'D:\selectImg\甘草', 0),
            (r'D:\selectImg\dealImg\chrysanthemum', 0),
            (r'D:\selectImg\丹参 穿孔病,叶斑病', 0),
            (r'D:\selectImg\地黄', 0),
            (r'D:\selectImg\防风', 0),
            (r'D:\selectImg\dealImg\mint', 0),
            (r'D:\selectImg\薄荷', 0),
            (r'D:\selectImg\dealImg\aster tataricus', 0),
            (r'D:\selectImg\dealImg\sanguisorba', 0)
            ]

meth = ['random_flip_horizon', 'random_flip_vertical', 'random_crop', 'gaussian_blur',
        'random_bright', 'random_contrast']


def file_filter(path):
    dataset = []
    all_file = os.listdir(path)
    for file in all_file:
        if not os.path.isdir(os.path.join(path, file)) and file.split('.')[1] == 'json':
            if os.path.exists(os.path.join(path, file.split('.')[0] + '.jpg')):
                dataset.append({'name': file.split('.')[0], 'path': path})
    return dataset


def json_to_txt(dataset):
    print('voc转yolo格式')
    dataset_return = []
    for i in tqdm(dataset):
        name = os.path.join(i['path'], i['name'] + '.json')
        save_name = os.path.join(i['path'], i['name'] + '.txt')
        if os.path.exists(save_name):
            os.remove(save_name)
        with open(name, 'r') as f:
            data = json.load(f)
            H = data.get('imageHeight')
            W = data.get('imageWidth')
            # classes = ["aster tataricus", "chrysanthemum", "digitalis purpurea", "glycyrrhiza uralensis",
            #            "mint", "paeonia", "salvia", 'sanguisorba', "saposhnikovia",
            #            "aster tataricus spot", "chrysanthemum spot", "digitalis purpurea spot",
            #            "glycyrrhiza uralensis spot",
            #            "mint spot", "paeonia spot", "salvia spot", 'sanguisorba spot', "saposhnikovia spot"]

            for k in data['shapes']:
                point_arr = []
                label_index = list(classes).index(str(k['label']))
                point_arr.append(str(label_index))
                points_contexts = (k['points'])
                if points_contexts[0][0] == points_contexts[1][0] or points_contexts[0][1] == points_contexts[1][1]:
                    continue
                x = round((points_contexts[0][0] + points_contexts[1][0]) / 2.0 / W, 6)
                y = round((points_contexts[0][1] + points_contexts[1][1]) / 2.0 / H, 6)
                w = round(abs(points_contexts[1][0] - points_contexts[0][0]) / W, 6)
                h = round(abs(points_contexts[1][1] - points_contexts[0][1]) / H, 6)
                line = ' '.join([str(label_index), str(x), str(y), str(w), str(h)]) + '\n'
                with open(save_name, 'a+') as ww:
                    ww.write(line)
        if os.path.exists(save_name):
            dataset_return.append({'name': i['name'], 'path': i['path']})
    return dataset_return


def check_dirs(res=res_path):
    if os.path.exists(res):
        shutil.rmtree(res)
    os.makedirs(res)
    os.makedirs(os.path.join(os.path.join(res, 'images'), 'train'))
    os.makedirs(os.path.join(os.path.join(res, 'images'), 'val'))
    os.makedirs(os.path.join(os.path.join(res, 'labels'), 'train'))
    os.makedirs(os.path.join(os.path.join(res, 'labels'), 'val'))


def split_dataset(dataset, res=res_path, train_num=0.8):
    count = 1

    flag = len(dataset) * train_num
    print('划分数据集')
    for i in tqdm(dataset):
        img_file = os.path.join(i['path'], i['name'] + '.jpg')
        label_file = os.path.join(i['path'], i['name'] + '.txt')
        if count <= flag:
            shutil.copy(img_file, os.path.join(os.path.join(os.path.join(res, 'images'), 'train'), i['name'] + '.jpg'))
            shutil.copy(label_file,
                        os.path.join(os.path.join(os.path.join(res, 'labels'), 'train'), i['name'] + '.txt'))
        else:
            shutil.copy(img_file, os.path.join(os.path.join(os.path.join(res, 'images'), 'val'), i['name'] + '.jpg'))
            shutil.copy(label_file,
                        os.path.join(os.path.join(os.path.join(res, 'labels'), 'val'), i['name'] + '.txt'))
        count += 1


if __name__ == '__main__':
    check_dirs(res_path)
    datasetList = []
    for path, enhanceTimes in pathList:
        if enhanceTimes == 0:
            datasetList.append(file_filter(path))
        else:
            datasetList.append(file_filter(path))
            if os.path.exists(os.path.join(path, 'create')):
                shutil.rmtree(os.path.join(path, 'create'))
            dataEnhance.create_datasets(method=meth, extimes=enhanceTimes, path=path)
            datasetList.append(file_filter(os.path.join(path, 'create')))
    for dataset in datasetList:
        split_dataset(json_to_txt(dataset))
