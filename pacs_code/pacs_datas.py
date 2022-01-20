import numpy as np 
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
# from utils import get_transform
import pdb
import random
import torch
import time

data_path = '../kfold/'

class PACS(Dataset):
	def __init__(self, test_domain, num_samples, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_samples = num_samples

		self.train_domain_imgs = []

		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				if int(label)-1 not in domain_imgs.keys():
					domain_imgs[int(label)-1] = []
				domain_imgs[int(label)-1].append(data_path + img)

			self.train_domain_imgs.append(domain_imgs)

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.val_img_list.append(data_path + img)
				self.val_label_list.append(int(label)-1)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../files/' + test_domain + '_test_kfold.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)
		# pdb.set_trace()

	def reset(self, phase, transform=None):
		# pdb.set_trace()
		if phase == 'meta_train':

			self.meta_test_domain = np.random.randint(len(self.domain_list))
			self.meta_train_imgs = []
			self.meta_train_labels = []
			self.transform = transform
			for j in range(len(self.train_domain_imgs[0])):
				for i in range(len(self.domain_list)):
					if i == self.meta_test_domain:
						continue
					else:
						img_names = self.train_domain_imgs[i][j]      # list i-th dictionary j-key
						random.shuffle(img_names)
						while len(img_names) < self.num_samples:
							img_names = img_names + img_names
						self.meta_train_imgs += img_names[:self.num_samples]   #shuffle & get first n samples
						self.meta_train_labels += [j] * self.num_samples
			self.img_list = self.meta_train_imgs
			self.label_list = self.meta_train_labels

		# generate meta_test imgs
		elif phase == 'meta_test':
			self.train_img_list = []
			self.train_label_list = []
			self.transform = transform
			meta_target = self.train_domain_imgs[self.meta_test_domain]  #dictionary
			for i in range(len(meta_target)):
				self.train_img_list += meta_target[i]
				self.train_label_list += [i] * len(meta_target[i])
			
			self.img_list = self.train_img_list
			self.label_list = self.train_label_list

		elif phase == 'all_train':
			self.meta_test_domain = np.random.randint(len(self.domain_list))
			self.all_train_imgs = []
			self.all_train_labels = []
			self.transform = transform

			for j in range(len(self.train_domain_imgs[0])):
				for i in range(len(self.domain_list)):
					img_names = self.train_domain_imgs[i][j]      # list i-th dictionary j-key
					random.shuffle(img_names)
					while len(img_names) < self.num_samples:
							img_names = img_names + img_names
					self.all_train_imgs += img_names[:self.num_samples]   #shuffle & get first n samples
					self.all_train_labels += [j] * self.num_samples
			self.img_list = self.all_train_imgs
			self.label_list = self.all_train_labels

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list
			self.label_list = self.val_label_list

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		image = image.resize((224, 224))
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		# return image and label
		return image, self.label_list[item]#, self.meta_train_imgs, self.meta_train_labels

	def __len__(self):
		return len(self.img_list)