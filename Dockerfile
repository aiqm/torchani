FROM zasdfgbnm/pytorch-master
COPY . /torchani
RUN cd torchani && python setup.py test
RUN cd torchani && python2 setup.py test
