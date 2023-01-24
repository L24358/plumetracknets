FROM belleliu/plumetracknets_intermediate
RUN pip3 install pytest
RUN pip3 install pysindy
RUN pip3 install tensorflow

COPY ./ /src

RUN pip3 install -e . 

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

RUN pip3 install -r requirements.txt