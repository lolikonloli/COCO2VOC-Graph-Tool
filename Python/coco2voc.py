from pycocotools.coco import COCO #这个包可以从git上下载https://github.com/cocodataset/cocoapi/tree/master/PythonAPI，也可以直接用修改后的coco.py
import  os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image

#Pyside2的三个路径


def CheckOsPath(path):      #检查文件夹是否已经存在
    if os.path.exists(path):        #存在，删除
        shutil.rmtree(path)
        os.mkdir(path)
    else:       #不存在，创建
        os.mkdir(path)


def save_annotations(filename, objs,filepath, voc_label_dir, voc_image_dir):
    annopath = voc_label_dir + "/" + filename[:-3] + "xml" #生成的xml文件保存路径
    dst_path = voc_image_dir + "/" + filename
    img_path=filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)#把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, img, classes, SImgFolderPath, voc_label_dir, voc_image_dir):
    filename = img['file_name']
    filepath=os.path.join(SImgFolderPath, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'],  iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
    save_annotations(filename, objs,filepath, voc_label_dir, voc_image_dir)


def catid2name(coco):#将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes


def get_CK5(SLabelFilePath, SImgFolderPath,voc_label_dir, voc_image_dir):
    coco = COCO(SLabelFilePath)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
    for imgId in tqdm(imgIds):
        img = coco.loadImgs(imgId)[0]
        showbycv(coco, img, classes, SImgFolderPath, voc_label_dir, voc_image_dir)


def Transform(SImgFolderPath, SLabelFilePath, TSaveFolderPath):        #转换函数
    coco2voc_full_dir = os.path.join(TSaveFolderPath, 'COCO2VOC')       #COCO2VOC文件夹位置
    CheckOsPath(coco2voc_full_dir)      #检查生成路径是否已经存在

    voc_label_dir = os.path.join(coco2voc_full_dir,'annotations')
    voc_image_dir = os.path.join(coco2voc_full_dir,'images')      #在上述文件夹中生成images，annotations两个子文件夹
    
    os.mkdir(voc_label_dir)
    os.mkdir(voc_image_dir)


    get_CK5(SLabelFilePath, SImgFolderPath, voc_label_dir, voc_image_dir)

if __name__ == '__main__':
    SImgFolderPath = r"E:\Paper-2020-1-lolikonloli\Data\pidray_transform\pidray\train"         #COCO图片文件夹绝对路径
    SLabelFilePath = r"E:\Paper-2020-1-lolikonloli\Data\pidray_transform\pidray\annotations\xray_train.json"         #COCO标签文件绝对路径
    TSaveFolderPath = r"E:\Paper-2020-1-lolikonloli\Data\pidray_transform"  
    Transform(SImgFolderPath, SLabelFilePath, TSaveFolderPath)