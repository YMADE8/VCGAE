'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pandas as pd
from utility.parser import parse_args
args = parse_args()

class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/trn_buy'
        # test_file = path + '/tst_buy'
        test_file = path + args.tst_file
        pv_file = path +'/trn_pv'
        fav_file = path +'/trn_fav'
        cart_file = path + '/trn_cart'

        #get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        self.exist_users = []
        self.train_items, self.test_set = {}, {}


        tp1 = pd.read_csv(train_file, sep=' ', names=['uid', 'iid'])
        tp1['uid'] -= 1
        tp1['iid'] -= 1
        tp1 = tp1.sort_values('uid')
        self.n_train = len(tp1)
        self.exist_users = list(np.unique(tp1['uid']))
        self.train_items = tp1.groupby('uid')['iid'].apply(list).to_dict()


        tp2 = pd.read_csv(test_file, sep=' ', names=['uid', 'iid'])
        tp2['uid'] -= 1
        tp2['iid'] -= 1
        tp2 = tp2.sort_values('uid')
        self.n_test = len(tp2)
        self.test_set = tp2.groupby('uid')['iid'].apply(list).to_dict()

        self.n_users = args.n
        self.n_items = args.m

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_pv = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_cart = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_fav = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.pv_set, self.fav_set,self.cart_set = {},{},{}

        for index in range(len(tp1)):
            uid = tp1.iloc[index][0]
            iid = tp1.iloc[index][1]
            self.R[uid, iid] = 1.


        tp3 = pd.read_csv(pv_file, sep=' ', names=['uid', 'iid'])
        tp3['uid'] -= 1
        tp3['iid'] -= 1
        tp3 = tp3.sort_values('uid')
        for index in range(len(tp3)):
            uid = tp3.iloc[index][0]
            iid = tp3.iloc[index][1]
            self.R_pv[uid, iid] = 1.
        self.pv_set = tp3.groupby('uid')['iid'].apply(list).to_dict()


        tp4 = pd.read_csv(cart_file, sep=' ', names=['uid', 'iid'])
        tp4['uid'] -= 1
        tp4['iid'] -= 1
        tp4 = tp4.sort_values('uid')
        for index in range(len(tp4)):
            uid = tp4.iloc[index][0]
            iid = tp4.iloc[index][1]
            self.R_cart[uid, iid] = 1.
        self.cart_set = tp4.groupby('uid')['iid'].apply(list).to_dict()

        tp5 = pd.read_csv(fav_file, sep=' ', names=['uid', 'iid'])
        tp5['uid'] -= 1
        tp5['iid'] -= 1
        tp5 = tp5.sort_values('uid')
        for index in range(len(tp5)):
            uid = tp5.iloc[index][0]
            iid = tp5.iloc[index][1]
            self.R_fav[uid, iid] = 1.
        self.fav_set = tp5.groupby('uid')['iid'].apply(list).to_dict()


    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')

            adj_mat_pv = sp.load_npz(self.path + '/s_adj_mat_pv.npz')
            norm_adj_mat_pv = sp.load_npz(self.path + '/s_norm_adj_mat_pv.npz')
            mean_adj_mat_pv = sp.load_npz(self.path + '/s_mean_adj_mat_pv.npz')

            adj_mat_fav = sp.load_npz(self.path + '/s_adj_mat_fav.npz')
            norm_adj_mat_fav = sp.load_npz(self.path + '/s_norm_adj_mat_fav.npz')
            mean_adj_mat_fav = sp.load_npz(self.path + '/s_mean_adj_mat_fav.npz')

            adj_mat_cart = sp.load_npz(self.path + '/s_adj_mat_cart.npz')
            norm_adj_mat_cart = sp.load_npz(self.path + '/s_norm_adj_mat_cart.npz')
            mean_adj_mat_cart = sp.load_npz(self.path + '/s_mean_adj_mat_cart.npz')

            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat(self.R)
            adj_mat_pv, norm_adj_mat_pv, mean_adj_mat_pv = self.create_adj_mat(self.R_pv)
            adj_mat_cart, norm_adj_mat_cart, mean_adj_mat_cart = self.create_adj_mat(self.R_cart)
            adj_mat_fav, norm_adj_mat_fav, mean_adj_mat_fav = self.create_adj_mat(self.R_fav)

            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)

            sp.save_npz(self.path + '/s_adj_mat_pv.npz', adj_mat_pv)
            sp.save_npz(self.path + '/s_norm_adj_mat_pv.npz', norm_adj_mat_pv)
            sp.save_npz(self.path + '/s_mean_adj_mat_pv.npz', mean_adj_mat_pv)

            sp.save_npz(self.path + '/s_adj_mat_cart.npz', adj_mat_cart)
            sp.save_npz(self.path + '/s_norm_adj_mat_cart.npz', norm_adj_mat_cart)
            sp.save_npz(self.path + '/s_mean_adj_mat_cart.npz', mean_adj_mat_cart)

            sp.save_npz(self.path + '/s_adj_mat_fav.npz', adj_mat_fav)
            sp.save_npz(self.path + '/s_norm_adj_mat_fav.npz', norm_adj_mat_fav)
            sp.save_npz(self.path + '/s_mean_adj_mat_fav.npz', mean_adj_mat_fav)
            
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            pre_adj_mat_pv = sp.load_npz(self.path + '/s_pre_adj_mat_pv.npz')
            pre_adj_mat_cart = sp.load_npz(self.path + '/s_pre_adj_mat_cart.npz')
            pre_adj_mat_fav = sp.load_npz(self.path + '/s_pre_adj_mat_fav.npz')

        except Exception:

            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            rowsum = np.array(adj_mat_pv.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat_pv)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre view adjacency matrix.')
            pre_adj_mat_pv = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_pv.npz', norm_adj)

            rowsum = np.array(adj_mat_cart.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat_cart)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre view adjacency matrix.')
            pre_adj_mat_cart = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_cart.npz', norm_adj)

            rowsum = np.array(adj_mat_fav.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat_fav)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre view adjacency matrix.')
            pre_adj_mat_fav = norm_adj.tocsr()
            sp.save_npz(self.path + '/s_pre_adj_mat_fav.npz', norm_adj)


        return pre_adj_mat,pre_adj_mat_pv,pre_adj_mat_cart,pre_adj_mat_fav

    def create_adj_mat(self,which_R):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = which_R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
    
        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
    
    
    
    
    
    
    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train)
        n_rates = 0

        split_state = []
        temp0=[]
        temp1=[]
        temp2=[]
        temp3=[]
        temp4=[]

        #print user_n_iid

        for idx, n_iids in enumerate(sorted(user_n_iid)):
            if n_iids <9:
                temp0+=user_n_iid[n_iids]
            elif n_iids <13:
                temp1+=user_n_iid[n_iids]
            elif n_iids <17:
                temp2+=user_n_iid[n_iids]
            elif n_iids <20:
                temp3+=user_n_iid[n_iids]
            else:
                temp4+=user_n_iid[n_iids]
            
        split_uids.append(temp0)
        split_uids.append(temp1)
        split_uids.append(temp2)
        split_uids.append(temp3)
        split_uids.append(temp4)
        split_state.append("#users=[%d]"%(len(temp0)))
        split_state.append("#users=[%d]"%(len(temp1)))
        split_state.append("#users=[%d]"%(len(temp2)))
        split_state.append("#users=[%d]"%(len(temp3)))
        split_state.append("#users=[%d]"%(len(temp4)))


        return split_uids, split_state



    def create_sparsity_split2(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
