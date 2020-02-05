import logging


ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)-15s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# log = logging.getLogger('DataSources')
# log.setLevel(logging.DEBUG)
# log.addHandler(ch)


FORMAT = '%(asctime)-15s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, handlers=[ch])
