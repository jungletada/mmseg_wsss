import argparse
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmsegmentation format')
    parser.add_argument('--devkit_path', default='data/VOCdevkit', help='pascal voc devkit path')
    args = parser.parse_args()
    return args


def main():
    s1 = len('/JPEGImages/')
    l = len('2011_003276')
    args = parse_args()
    devkit_path = args.devkit_path
    
    with open(osp.join(devkit_path, 'VOC2012/ImageSets/SegmentationAug', 'train_aug.txt')) as f:
        train_aug_list = [line.strip() for line in f]
    
    f = open(osp.join(devkit_path, 'VOC2012/ImageSets/SegmentationAug', 'train_aug_id.txt'),
             'w')
    train_aug_id = [line[s1:s1+l] for line in train_aug_list]
    f.writelines(line + '\n' for line in train_aug_id)
    f.close()
    
    print('Done!')


if __name__ == '__main__':
    main()