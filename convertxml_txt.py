import xml.etree.ElementTree as ET
import os

VOC_CLASSES = (    # always index 0
    'person','hat')

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    # print(tree)
    objects = []
    
    for imgtest in tree.findall('size'):
        width = float(imgtest.find('width').text)
        height = float(imgtest.find('height').text)
        depth = float(imgtest.find('depth').text)
    
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        # print(obj_struct['name'])

        obj_struct['bbox'] = [round(float(bbox.find('xmin').text)/width,3),
                              round(float(bbox.find('ymin').text)/height,3),
                              round(float(bbox.find('xmax').text)/width,3),
                              round(float(bbox.find('ymax').text)/height,3)]
        objects.append(obj_struct)

    return objects


Annotations = './label/'
xml_files = os.listdir(Annotations)
#print(xml_files)
# results = parse_rec(Annotations + "new_1.xml")

txt_file = open('train.txt','w')
count = 0
for xml_file in xml_files:
    count += 1
    results = parse_rec(Annotations + xml_file)
    if len(results)==0:
        # print(xml_file)
        continue
    image_path = xml_file[4:-4] + ".jpg"
    # print(image_path)
    img_root = os.getcwd()
    image_path = os.path.join(img_root,"train",image_path)
    txt_file.write(image_path)

    for result in results:
        class_name = result['name']
        if class_name not in VOC_CLASSES:
            continue
        # print(class_name)
        bbox = result['bbox']
        class_name = VOC_CLASSES.index(class_name)
        txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
    txt_file.write('\n')
    #if count == 10:
    #    break
txt_file.close()
