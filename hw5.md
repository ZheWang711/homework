# CSE250B homework5
Zhe Wang
A53097553

## Question1
* (a): The boundary is not linear!
![](/Users/zhewang711/Downloads/a)
* (b) The key idea here is to repalce the *w* whose corresponding *c* is smallest whenever a new *w* comes in!
```Python
def voted_perceptron_dsp(X, Y, T, L):
    n = len(X)
    l = 0 # l is the index of last w
    c = np.array([0])
    w = np.array([[0, 0, 0]])
    data_map = np.arange(n)
    for _ in range(T):
        np.random.shuffle(data_map)
        for i in range(n):
            if np.sign(np.dot(w[l], X[data_map[i]])) != Y[data_map[i]]:
                if len(w) == L:
                    nxt = np.argmin(c) # nxt is the index of next w
                    w[nxt] = np.array([w[l] + Y[data_map[i]]*X[data_map[i]]])
                    l = nxt
                else:
                    w = np.append(w, [w[l] + Y[data_map[i]]*X[data_map[i]]], axis=0)
                    c = np.append(c, 1)
                    l += 1
            else:

                c[l] = c[l] + 1

    return w, c
```
Boundary after 1000000 iterations
![](/Users/zhewang711/Downloads/b)

* ( c)
Here we can use the idea similar with dynamic programming (remembering the latest result and take advantage it when computing new w values)
```Python
w1 = 0 # w1 is current wl
w2 = 0 # w2 is w that is going to be used in classification
Repeat T times:
    Randomly permute the data points
    for i from 1 to n:
        if (x(i), y(i)) is misclassified by w1:
            w1 = w1 + y(i)x(i)
            w2 = w2 + w1
        else:
            w2 = w2 + w1
return w2
```
Result:
![](/Users/zhewang711/Downloads/c)

## Question2
* quadratic kernel:
    * data1.txt
    ![](/Users/zhewang711/Downloads/2-1-1)
    * data2.txt
    ![](/Users/zhewang711/Downloads/2-1-2)
* RBF kernel:
    * data1.txt
        * sigma = 1
        ![](/Users/zhewang711/Downloads/2-2-data1-1)
        * sigma = 10
        ![](/Users/zhewang711/Downloads/2-2-data1-10)
        * sigma = 20
        ![](/Users/zhewang711/Downloads/2-2-data1-20)
    * data2.txt
        * sigma = 1
        ![](/Users/zhewang711/Downloads/2-2-data2-1)
        * sigma = 10
        ![](/Users/zhewang711/Downloads/2-2-data2-10)
        * sigma = 20
        ![](/Users/zhewang711/Downloads/2-2-data2-20)
