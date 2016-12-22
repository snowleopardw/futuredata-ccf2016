import os
import sys
import csv
import pdb

"""
@author: Aaron_Huang
"""
def get_filename(rootDir):
	"""
    获取文件名称

    Args:
        rootDir: 根目录
    Returns:
        result: 返回一个list，记录所有文件的文件名
    """
	finalresult = [] 
	list_dirs = os.walk(rootDir) 
	for root, dirs, files in list_dirs: 
		for f in files: 
			filename = os.path.join(root, f).split('\\')[-1]
			withoutjpg = filename.split('.')[0]
			finalresult.append([withoutjpg])
	return finalresult


if __name__ == '__main__':
	rootDir = sys.argv[1]
	finalresult = get_filename('./'+rootDir)
	with open(rootDir+'.csv', 'w',newline='',encoding='utf-8') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',')
		# pdb.set_trace()
		spamwriter.writerows(finalresult)