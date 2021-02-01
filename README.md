# Тестовое задание MIL

Обучить ResNet20 на CIFAR10, реализовать метод pruning'а и оценить его результаты.

### Выполнение задания

**Шаг 1.** Pruning  
Метод снижения сложности нейронной сети, через удаление наименее важных весов модели, который позволяет без значительной потери качества уменьшить ее размер, а значит и требуемые для использования ресурсы.  
Filter level pruning - метод удаления низкоэффективных фильтров.  
В данном задании, будем заменять фильтры на центроиды кластеров полученных с помощью KMeans.  

**Шаг 2.** Датасет CIFAR10.  
Задание будем выполнять на датасете CIFAR10 - состоит из 60000 цветных изображений 32х32 в 10 классах (по 6000 на каждый класс). В нем 50000 тренировночных картинок и 10000 тестовых.

**Шаг 3.** ResNet20.  
Архитектуар реализована в [res_net.py](https://github.com/veseliy/mil_task/blob/main/res_net.p).

**Шаг 4.** Обучение.  
Скрипт обучения [train_resnet.py](https://github.com/veseliy/mil_task/blob/main/train_resnet.py)  

Попробовал разное кол-во эпох от 10 до 1000, но необходимого результата в 90% добиться не получилось.  

Лучший результат на 100 эпохах - 81% точности  

Accuracy of plane : 79 %  
Accuracy of   car : 92 %  
Accuracy of  bird : 74 %  
Accuracy of   cat : 57 %  
Accuracy of  deer : 75 %  
Accuracy of   dog : 82 %  
Accuracy of  frog : 86 %  
Accuracy of horse : 87 %  
Accuracy of  ship : 91 %  
Accuracy of truck : 87 %  

**Шаг 5.** Filter-level метод pruning'а.  
Функция и ее исполнение [prunning.py](https://github.com/veseliy/mil_task/blob/main/prunning.py)  
В ней не вырезал ненужные фильтры - а просто заменил веса в каждом фильтре на центроиды кластера, те у меня получилась сеть с повторяющимися фильтрами.  
Таким образом не пришлось менять структуру сети и получилось быстро провести эксперимент. В дальнейшем можно изменить структуру сети и таким образом снизить кол-во параметров.  

Успел попробовать только 1 вариант:  
Кол-во кластеров во всех свертках одинаковое и равно - 3.  
Это дало следующие изменения в точности относительно базового варианта.  

Accuracy of plane : +2 %  
Accuracy of   car : -1 %  
Accuracy of  bird : -3 %  
Accuracy of   cat : -3 %  
Accuracy of  deer : -4 %  
Accuracy of   dog : -2 %  
Accuracy of  frog : 0 %  
Accuracy of horse : -4 %  
Accuracy of  ship : -2 %  
Accuracy of truck : -1 %  

В среднем ухудшение на 1.8%.  

**Шаг 6.** Визуализировать результаты и провести их анализ.  

Данный этап реализовать не успел  

**Шаг 7.** Сформулировать идеи для улучшения текущих результатов.  

Обучение:  
1. Проверить архитектур  
2. Проверить инициализацию весов  
3. Внедрить плавающий lr  

Pruning:  
1. Пробовать разное кол-во кластеров для разных сверток (как было указано в задании).  
2. Пробовать разные метрики кластеризации.  
  
