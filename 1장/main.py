class Man:
    def __init__(self,name): # 클래스 초기화 __init__ 은 초기화용 매서드이고 생성자라고 한다.
        self.name = name # 파이썬에서 매서드의 첫 번쨰 인수로 자신을 나타내는 self를 씀
        print('Initialized!')
    
    def hello(self):
        print('Hello ' + self.name + '!')

    def goodbye(self):
        print('Good-bye ' + self.name + '!')

m = Man('David')
m.hello()
m.goodbye()