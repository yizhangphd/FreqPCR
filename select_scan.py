import shutil
import os

scene = 'data/scannet/0045'
source_img = os.path.join(scene, 'color')
targer_img = os.path.join(scene, 'color_select')
source_pose = os.path.join(scene, 'pose')
target_pose = os.path.join(scene, 'pose_select')

os.mkdir(targer_img)
os.mkdir(target_pose)
total = len(os.listdir(source_img))

step = min(total//100, 20)
print('step:', step)

if scene == 'data/scannet/0043':
    total = 1000
    step = 10

for i, j in enumerate(range(0, total, step)):
    print(j)
    shutil.copy(f'{source_img}/{j}.jpg', f'{targer_img}/{i}.jpg')
    shutil.copy(f'{source_pose}/{j}.txt', f'{target_pose}/{i}.txt')

print('OK')