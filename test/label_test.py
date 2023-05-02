import unittest 
import os 
import glob
import json

class TestLabel(unittest.TestCase):
    def setUp(self):
        label_name = 'multi_label'
        dataset_prefix = './dataset/' 
        self.obj_count = 4
        self.obj_len = 4
        self.label_path_list = glob.glob(os.path.join(dataset_prefix,label_name,'*'))

    def test_Label_name(self):
        cats = set()
        for label_path in self.label_path_list:
            with open(label_path , 'r') as file:
                json_file = json.load(file)

            obj_list = json_file['shapes']
            self.assertEqual(len(obj_list) , self.obj_count)
            for obj in obj_list:
                label = obj['label']
                cats.add(label)

        print(cats)
        self.assertEqual(len(cats) , self.obj_len)


if __name__=='__main__':
    unittest.main()