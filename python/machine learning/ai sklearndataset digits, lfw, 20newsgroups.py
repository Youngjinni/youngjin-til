from sklearn import datasets
import matplotlib.pyplot as plt

digit = datasets.load_digits() # load digit datasets

plt.figure(figsize=(5,5)) # figure number image 5, 5
plt.imshow(digit.images[0], cmap=plt.cm.gray_r,interpolation = 'nearest')

plt.show()
print(digit.data[0])
print('이 숫자는 ', digit.target[0], '입니다.')

lfw = datasets.fetch_lfw_people(min_faces_per_person=70,resize=0.4) # load flw dataset

plt.figure(figsize=(20,5)) # figure people image 20, 5

for i in range(8):
    plt.subplot(1,8,i+1)
    plt.imshow(lfw.images[i],cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])
    
plt.show()

news = datasets.fetch_20newsgroups(subset = 'train')
print('*****\n', news.data[0], '\n*****')
print('이 문서의 부류는 <', news.target[0], '>입니다.')