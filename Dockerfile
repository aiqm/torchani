FROM zasdfgbnm/pytorch-master
RUN pacman -Sy --noconfirm python-sphinx python2-sphinx
COPY . /torchani
RUN cd torchani && python setup.py test
#RUN cd torchani && python2 setup.py test
