import tensorflow as tf
import os
import glob
import abc

# Configuration class.
class Config(object):
	LOG_DEVICE_PLACEMENT = False
	IMG_SIZE = 256
	MAP_SIZE = 32
	FIG_SIZE = 128	

	# Training meta.
	STEPS_PER_EPOCH = 2000
	IMG_LOG_FR = 1000
	TXT_LOG_FR = 1000
	# Epochs after which learning rate decays.
	NUM_EPOCHS_PER_DECAY = 10.0   

	# Initial learning rate.
	lr = 1e-4
	# Learning rate decay factor.       
	LEARNING_RATE_DECAY_FACTOR = 0.89  
	# The decay to use for the moving average.
	LEARNING_MOMENTUM = 0.999  
	# The decay to use for the moving average.   
	MOVING_AVERAGE_DECAY = 0.9999     
	GAN = 'ls' # 'hinge', 'ls'
	DECAY_STEP = 1

	# Discriminator depth.
	n_layer_D = 4

	def __init__(self, args):
		self.MAX_EPOCH = args.epoch
		self.GPU_INDEX = args.cuda
		self.phase = args.stage
		assert self.phase in ['pretrain', 'ft', 'ub'], print("Please offer the valid phase!")
		self.type  = args.type
		self.SET = args.set
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				tf.config.experimental.set_memory_growth(gpus[self.GPU_INDEX], True)
				tf.config.experimental.set_visible_devices(gpus[self.GPU_INDEX], 'GPU')
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
			except RuntimeError as e:
				print(e) # Virtual devices must be set before GPUs have been initialized

	@abc.abstractmethod
	def search_folder(self, root_dir, sub_id, stype):
		# pass
		return

	@abc.abstractmethod
	def search_folder_wrapper(self, root_dir, filenames):
		# pass
		return

	def compile(self):
		protocol_type = self.SET

		# Training data.
		self.SP_DATA_DIR, self.LI_DATA_DIR = [], []
		counter = 0
		if self.phase == 'pretrain':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
		elif self.phase == 'ft':
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
		elif self.phase == 'ub':
			with open(self.pretrain_train, 'r') as f:
				filenames0 = f.read().split('\n')	
			with open(self.type_train, 'r') as f:
				filenames1 = f.read().split('\n')
			filenames = filenames0 + filenames1

		self.LI_DATA_DIR, self.SP_DATA_DIR = self.search_folder_wrapper(self.root_dir, 
																		 filenames)
		with open(self.pretrain_test, 'r') as f:
			filenames = f.read().split('\n')
		self.LI_DATA_DIR_TEST, self.SP_DATA_DIR_TEST = self.search_folder_wrapper(self.root_dir, 
																				   filenames)
		with open(self.type_test, 'r') as f:
			filenames = f.read().split('\n')
		self.LI_DATA_DIR_TEST_B, self.SP_DATA_DIR_TEST_B = self.search_folder_wrapper(self.root_dir,
																						filenames)

	def filename_gen(self):
		# TODO: one function finds the filenames.
		return 

class Config_oulu(Config):

	def __init__(self, args):
		super().__init__(args)
		self.dataset = 'oulu'
		self.BATCH_SIZE = 2
		self.root_dir = "/user/guoxia11/cvlshare/Databases/Oulu/bin/"
		root_dir_id = '/user/guoxia11/cvl/anti_spoofing/stats_update/oulu_datalist/'	
		self.pretrain_train = root_dir_id + 'A_train_oulu.txt'
		self.pretrain_test  = root_dir_id + 'A_test_oulu.txt' 

		if self.type == 'age':
			self.type_train = root_dir_id + 'C_train_oulu.txt'
			self.type_test  = root_dir_id + 'C_test_oulu.txt'
		elif self.type == 'spoof':
			self.type_train = root_dir_id + 'B_train_oulu.txt'
			self.type_test  = root_dir_id + 'B_test_oulu.txt'
		elif self.type == 'race':
			self.type_train = root_dir_id + 'D_train_oulu.txt'
			self.type_test  = root_dir_id + 'D_test_oulu.txt'
		else:
			assert False, print("wait to implement...")

	def search_folder(self, root_dir, sub_id, stype):
		super(Config_oulu, self).search_folder(root_dir, sub_id, stype)
		if stype == 'Live':
			folder_list = glob.glob(root_dir+f'train/live/*{sub_id}*')
			folder_list += glob.glob(root_dir+f'eval/live/*{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/live/*{sub_id}*')
		elif stype == 'Spoof':
			folder_list = glob.glob(root_dir+f'train/spoof/*{sub_id}*')		
			folder_list += glob.glob(root_dir+f'eval/spoof/*{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/spoof/*{sub_id}*')
		else:
			assert False, print("Please offer a valid stype here.")
		return folder_list

	def search_folder_wrapper(self, root_dir, filenames):
		super(Config_oulu, self).search_folder_wrapper(root_dir, filenames)
		li_list, sp_list = [], []
		for x in filenames:
			if x in ["0", ""]:
				continue
			else:
				digit_len = len(x)
				if digit_len == 1:
					sub_id = '0'+x
				else:
					sub_id = x
				li_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Live")
				sp_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Spoof")
		return li_list, sp_list

class Config_siw(Config):

	def __init__(self, args):
		super().__init__(args)
		self.dataset = "SiW"
		self.BATCH_SIZE = 2
		root_dir_id = '/user/guoxia11/cvl/anti_spoofing/stats_update/SiW_datalist/'
		self.root_dir = "/user/guoxia11/cvlshare/Databases/SiW/bin/"
		self.pretrain_train = root_dir_id + 'A_train_sub_id_siw.txt'
		self.pretrain_test  = root_dir_id + 'A_test_sub_id_siw.txt' 

		if self.type == 'age':
			self.type_train = root_dir_id + 'C_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'C_test_sub_id_siw.txt'
		elif self.type == 'spoof':
			self.type_train = root_dir_id + 'B_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'B_test_sub_id_siw.txt'
		elif self.type == 'race':
			self.type_train = root_dir_id + 'D_train_sub_id_siw.txt'
			self.type_test  = root_dir_id + 'D_test_sub_id_siw.txt'
		else:
			assert False, print("wait to implement...")

	def search_folder(self, root_dir, sub_id, stype):
		super(Config_siw, self).search_folder(root_dir, sub_id, stype)
		if stype == 'Live':
			folder_list = glob.glob(root_dir+f'train/live/{sub_id}*')
			folder_list += glob.glob(root_dir+f'test/live/{sub_id}*')
		elif stype == 'Spoof':
			folder_list = glob.glob(root_dir+f'train/spoof/{sub_id}*')		
			folder_list += glob.glob(root_dir+f'test/spoof/{sub_id}*')
		else:
			assert False, print("Please offer a valid stype here.")
		return folder_list

	def search_folder_wrapper(self, root_dir, filenames):
		super(Config_siw, self).search_folder_wrapper(root_dir, filenames)
		li_list, sp_list = [], []
		for x in filenames:
			if x in ["0", ""]:
				continue
			else:
				digit_len = len(x)
				if digit_len == 1:
					sub_id = '00'+x
				elif digit_len == 2:
					sub_id = '0'+x
				else:
					sub_id = x
				li_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Live")
				sp_list += self.search_folder(root_dir=root_dir, sub_id=sub_id, stype="Spoof")
		return li_list, sp_list

class Config_siwm(Config):

	def __init__(self, args):
		super().__init__(args)
		self.dataset = "SiWM-v2"
		self.BATCH_SIZE = 4
		root_dir_id = "/user/guoxia11/cvl/anti_spoofing/"
		if self.type == 'age':
			self.type_train = root_dir_id + "age_list/list/age_B_train_ub.txt"
			self.type_test  = root_dir_id + "age_list/list/age_B_test.txt"
		elif self.type == 'spoof':
			self.type_train = root_dir_id + "spoof_type_list/B_train_spoof_balanced_ub.txt"
			self.type_test  = root_dir_id + "spoof_type_list/B_test_spoof.txt"
		elif self.type == 'race':
			self.type_train = root_dir_id + "race_list/race_small_B_train_ub.txt"
			self.type_test  = root_dir_id + "race_list/race_B_test.txt"
		else:
			assert False, print("wait to implement...")
		self.pretrain_train = root_dir_id + 'spoof_type_list/pretrain_A_train_balanced.txt'
		self.pretrain_test  = root_dir_id + 'spoof_type_list/pretrain_A_test.txt'

	def compile_siwm(self):
		# Training data.
		self.SP_DATA_DIR, self.LI_DATA_DIR = [], []
		if self.phase == 'pretrain':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')
		elif self.phase == 'ft':
			with open(self.type_train, 'r') as f:
				filenames = f.read().split('\n')
		elif self.phase == 'ub':
			with open(self.pretrain_train, 'r') as f:
				filenames = f.read().split('\n')	
			with open(self.type_train, 'r') as f:
				filenames += f.read().split('\n')

		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)

		# Val/Test data.
		with open(self.pretrain_test, 'r') as f:
			filenames = f.read().split('\n')
		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR_TEST.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR_TEST.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)

		# Val/Test data.
		with open(self.type_test, 'r') as f:
			filenames = f.read().split('\n')
		for x in filenames:
			if x == '':
				continue
			elif 'Live' not in x:
				self.SP_DATA_DIR_TEST_B.append('/user/guoxia11/cvlshare/cvl-guoxia11/Spoof/'+x)
			else:
				self.LI_DATA_DIR_TEST_B.append('/user/guoxia11/cvlshare/cvl-guoxia11/Live/'+x)
