import numpy as np
import io_tool as it
import os

if __name__ == '__main__':
    # test_instrument = 'Accordion'
    # test_instrument = 'Cello'
    # test_instrument = 'Drummer'
    # test_instrument = 'Flute'
    # test_instrument = 'Guitar'
    # test_instrument = 'Piano'
    # test_instrument = 'Trumpet'
    # test_instrument = 'Violin'

    # Data options
    base_dir = "/home/ciaua/NAS/home/data/youtube8m/"

    # Tag
    tag_fp = os.path.join(base_dir, 'tag_list.instrument.csv')
    tag_list = [term[0] for term in it.read_csv(tag_fp)]

    anno_dir = os.path.join(base_dir, 'annotation', 'validate.500')
    fn_list = os.listdir(anno_dir)

    idict = {tag: [] for tag in tag_list}

    for fn in fn_list:
        anno_fp = os.path.join(anno_dir, fn)
        anno = np.load(anno_fp)
        nonzero = anno.nonzero()[0]
        for idx in nonzero:
            idict[tag_list[idx]].append(fn.replace('.npy', ''))
