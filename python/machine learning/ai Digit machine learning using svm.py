from sklearn import datasets
from sklearn import svm

digit = datasets.load_digits()

s = svm.SVC(gamma = 0.1, C= 10)
s.fit(digit.data, digit.target)

new_d = [digit.data[0], digit.data[1], digit.data[2]]
res = s.predict(new_d)
print('예측값은 ', res)
print('참값은 ', digit.target[0], digit.target[1], digit.target[2])

res = s.predict(digit.data)
correct = [i for i in range(len(res)) if res[i] == digit.target[i]]
accuracy = len(correct)/len(res)
print('정답률은 ', accuracy * 100, '%입니다.')
