#ifndef CORETREELABELLING_H_
#define CORETREELABELLING_H_

#include <omp.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <queue>
#include <bitset>
#include <cstring>
#include <algorithm>
#include <utility>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <climits>
#include <iostream>
#include <functional>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <algorithm>
#include <cmath>

using namespace std;
#define MAXN 1000000000
#define MAXINT ((unsigned)4294967295)
#define MAXGID 100000000
#define MAXLINE 1024
#define MAXT 65000
#define MAXD 120
#define POSINF 2147483647
#define NEGINF 0

#define N_ROOTS 4
#define MAX_BP_THREADS 8

#define RANK_STATIC 0
#define RANK_LOCAL_STATIC 1	   // LS
#define RANK_HOP_BETWEENNESS 2 // HB

typedef unsigned short tint;
typedef pair<int, int> pii;
typedef char dint;

vector<int> score, _score;

class VAL
{
public:
	VAL(int id, int val);
	int id, val;
	bool operator<(const VAL &v) const;
};
bool VAL::operator<(const VAL &v) const
{
	return val == v.val ? id < v.id : val > v.val;
}

VAL::VAL(int id, int val)
{
	this->id = id;
	this->val = val;
}

class DIS
{
public:
	DIS() : id(0), dis() {}
	DIS(int id, vector<int> dis);
	int id;
	vector<int> dis;
	bool operator<(const DIS &v) const;
	void clear()
	{
		dis.clear();
		id = 0;
	}
};

DIS::DIS(int id, vector<int> dis) : id(id), dis(dis) {}

bool DIS::operator<(const DIS &v) const
{
	return id < v.id;
}

class IntVal
{
public:
	int x;

public:
	IntVal();
	IntVal(int _x);
	bool operator<(const IntVal &v) const;
};

class DV
{
public:
	DV();
	DV(int id, double val);
	int id;
	double val;
	bool operator<(const DV &v) const;
};

class CM
{
public:
	CM() : id(0) {}
	CM(int id, int fir);
	int id;
	vector<int> ve;
};

class Vertex
{
public:
	vector<int> adj;
	int lb, ub, len;
};

class Val
{
public:
	Val(int id, int val);
	int id, val;
	bool operator<(const Val &v) const;
};
bool Val::operator<(const Val &v) const
{
	return val == v.val ? id < v.id : val > v.val;
}
Val::Val(int id, int val)
{
	this->id = id;
	this->val = val;
}

struct MEM
{
	int cur, source, dis;
	bool operator<(const MEM &other) const
	{
		return dis > other.dis;
	}
};

double get_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

class TreeNode
{
public:
	int id, f, h, rid, rsize, w, ch_n;
	vector<int> nbr, cost; // of size <=w
	vector<int> anc;	   // of size h
	vector<dint> dis;
	vector<int> ch;
};

class CoreTree
{
public:
	static inline bool get_edge(char *line, int &a, int &b, int &c, int num_cnt = 2);
	static inline int get_num_cnt(string path);
	static void create_bin(string path, string graph_name, int rank_threads = 1,
						   int rank_method = RANK_STATIC, int rank_max_minutes = 1000000, int max_hops = 3, bool merge_equv = true, string bin_path = "");
	static void get_order(vector<pair<int, int>> *con, int n, int *o, int method);

public:
	string path;
	int **con, *dat, *deg, *nid, *old, *old_new, *new_old, *f;
	int n_org, n, n_core;
	long long m, m_core;
	int root_n;
	int core_num = 0, branch_num = 0;

public:
	CoreTree(string path);
	~CoreTree();
	void load_graph(string bin_path);

public:
	int id;
	double t;
	int max_w;
	int numLand;
	int stamp_vis, stamp_dis, stamp;
	long long nnnn;
	double t0_sum = 0, t1_sum = 0, t2_sum = 0, t3_sum = 0, t4_sum = 0, t5_sum = 0;
	vector<int> d_l;
	vector<vector<int>> nbr, cost;
	vector<int> ord, rank;
	vector<int> root;
	vector<int> root_to_idex;
	vector<vector<pair<int, int>>> E;
	vector<TreeNode> tree;
	vector<Vertex> V;
	vector<vector<int>> nodeGroup;
	vector<int> landId;
	vector<vector<int>> disLand;
	vector<vector<Val>> listLand;
	vector<int> father;
	vector<vector<DIS>> root_dis;
	vector<vector<DIS>> root_visited;
	vector<int> candidate;
	vector<vector<int>> weight;
	vector<vector<vector<int>>> sum_update_dis;
	vector<int> vi;
	vector<int> id_to_index;
	vector<int> stamp_d;
	vector<int> stamp_v;
	vector<int> Dijsktra_dis;
	vector<vector<int>> Muti_Dijsktra_dis;
	vector<vector<int>> Muti_Dijsktra_vis;
	vector<vector<int>> update_dis;
	vector<vector<int>> ListToIndex;
	vector<int> node_g;
	vector<int> group_l;
	vector<int> group_r;
	vector<int> group_len;
	vector<int> group_size;
	vector<vector<int>> group_d;
	vector<vector<int>> group_mem;
	vector<vector<int>> Dij_dis;
	vector<int> record_v;
	vector<int> used_v;
	vector<int> curGap;
	queue<pair<int, int>> H;

public:
	// tree
	void reduce(int max_w, int n_threads);
	void decompose_tree(int max_w, int n_threads, string bin_path);
	void create_tree();
	void compute_tree_label();
	void compute_tree_label(int x, int rsize, vector<TreeNode *> &s);

	// load
	void load_label(int max_w, string bin_path, int lambmark);
	void load_label_tree(int max_w, string bin_path);
	void save_label_tree(int max_w, string bin_path);
	void save_tmp_graph(int max_w, string bin_path);
	void load_tmp_graph(int max_w, string bin_path);

	// reference
	void init();
	void create_landmark();
	void select_landmark();
	void distance_landmark();
	void update_bound();
	void assign_group();

	// update_bound
	inline void update_ub(int v, int ub);
	inline void update_lb(int v, int lb);

	// get_result
	int get_unmch(int lid);
	int get_unmch();

	// compute_ecc
	void compute_ecc(bool flag);
	void compute_ecc(int lid, int &lambcnt, int &cnt, vector<int> &visited, vector<int> &node_vis, vector<int> &node_continue, int &max_id, int &max_lb, bool flag);
	void compute_ecc1(int lid, int &lambcnt, int &cnt, vector<int> &visited, vector<int> &node_vis, vector<int> &node_continue, int &max_id, int &max_lb, bool flag);
	int binarySearch(const vector<int> &arr, int target, int g_len, int d1, int d2);
	void Muti_Dijsktra(int u);

	// save
	void txt_to_file(string graph_name);

public:
	pair<int, int> **core_con;
	pair<int, int> *core_dat;
	int *core_deg;

public:
	tint *last_t;
	int *dis;
	tint nowt;

public:
	// calculate
	void init_query();
	void Dijkstra(int u, vector<int> &distance);
	bool Dijkstra_Judge(int u, int lid, int lambda, vector<int> &node_vis, vector<int> &node_continue);
	bool Dijkstra_Judge1(int u, vector<int> &node_vis, vector<int> &node_continue);
	void Local_Dijkstra(int u, int stamp);
	void Local_Dijkstra(int u, int stamp, vector<vector<int>> &up_dis);
	void Local_Dij(int u, int lambda, int lid, bool flag);
	void Compute_root_dis(int u);
	void Merge_root_dis();
	void Compute_group();
	bool check(int u, vector<int> &visited, bool flag);
};

bool DV::operator<(const DV &v) const
{
	if (val > v.val + 1e-8)
		return true;
	if (val < v.val - 1e-8)
		return false;
	return id < v.id;
}

DV::DV(int id, double val)
{
	this->id = id;
	this->val = val;
}

DV::DV()
{
	id = -1;
	val = -1;
}

IntVal::IntVal() { x = -1; }
IntVal::IntVal(int x) { this->x = x; }
bool IntVal::operator<(const IntVal &v) const
{
	if (score[x] == score[v.x])
		return x < v.x;
	return score[x] < score[v.x];
}

CoreTree::CoreTree(string path)
{
	deg = NULL;
	dat = NULL;
	con = NULL;
	nid = NULL;
	old = NULL;
	f = NULL;
	n_org = 0;
	n = 0;
	m = 0;
	n_core = 0;
	m_core = 0;
	max_w = 0;
	last_t = NULL;
	dis = NULL;
	nowt = 0;
	core_con = NULL;
	core_dat = NULL;
	this->path = path;
	t = omp_get_wtime();
	core_deg = NULL;
}

CoreTree::~CoreTree()
{
	if (deg)
		delete[] deg;
	if (dat)
		delete[] dat;
	if (con)
		delete[] con;
	if (nid)
		delete[] nid;
	if (old)
		delete[] old;
	if (f)
		delete[] f;
	if (last_t)
		delete[] last_t;
	if (dis)
		delete[] dis;
	if (core_con)
		delete[] core_con;
	if (core_dat)
		delete[] core_dat;
	if (core_deg)
		delete[] core_deg;
}

void CoreTree::init_query()
{
	nowt = 0;
	last_t = new tint[n_core];
	memset(last_t, 0, sizeof(tint) * n_core);
	dis = new int[n_core];
}

void CoreTree::Dijkstra(int u, vector<int> &distance)
{
	priority_queue<pii, vector<pii>, greater<pii>> pq;
	vector<int> vis;
	vis.assign(n, 0);
	pq.push({0, u});
	distance.clear();
	distance.assign(n, INT_MAX);

	distance[u] = 0;
	while (!pq.empty())
	{
		int curDist = pq.top().first;
		int cur = pq.top().second;
		pq.pop();

		if (vis[cur])
		{
			continue;
		}
		vis[cur] = 1;

		for (int i = 0; i < deg[cur]; i++)
		{
			int v = con[cur][i];
			int d = weight[cur][i] + curDist;
			if (d < distance[v])
			{
				distance[v] = d;
				pq.push({distance[v], v});
			}
		}
	}
}
bool CoreTree::Dijkstra_Judge(int u, int lid, int lambda, vector<int> &node_vis, vector<int> &node_continue)
{
	priority_queue<MEM> pq;
	vector<int> nbr;
	int Dij_size = 0;
	bool flag_chain;
	stamp_vis++;
	stamp_dis++;
	int group_index = node_g[u];
	int left = group_l[group_index], right = group_r[group_index];
	if (deg[u] == 2)
	{
		if (node_continue[left] == 1 && node_continue[right] == 1)
			return true;
		if (!node_continue[left])
		{
			pq.push(MEM{left, Dij_size, 0});
			Dij_dis[left].resize(n);
			Dij_dis[left][left] = 0;
			Muti_Dijsktra_dis[Dij_size][left] = stamp_dis;
			nbr.push_back(left);
			Dij_size++;
		}
		if (!node_continue[right] && left != right)
		{
			pq.push(MEM{right, Dij_size, 0});
			Dij_dis[right].resize(n);
			Dij_dis[right][right] = 0;
			Muti_Dijsktra_dis[Dij_size][right] = stamp_dis;
			nbr.push_back(right);
			Dij_size++;
		}
		flag_chain = true;
		left = group_mem[group_index].front(), right = group_mem[group_index].back();
	}
	else
	{
		pq.push(MEM{u, 0, 0});
		Dij_dis[u].resize(n);
		Dij_dis[u][u] = 0;
		Muti_Dijsktra_dis[Dij_size][u] = stamp_dis;
		nbr.push_back(u);
		flag_chain = false;
		Dij_size++;
	}
	int aa = 0;
	while (!pq.empty())
	{
		nnnn++;
		MEM member = pq.top();
		int cur = member.cur;
		int source = member.source;
		int nowDis = member.dis;
		pq.pop();

		if (Muti_Dijsktra_vis[source][cur] == stamp_vis)
		{
			continue;
		}
		Muti_Dijsktra_vis[source][cur] = stamp_vis;

		if (flag_chain == false)
		{
			if (curGap[cur] != 0)
			{
				update_lb(cur, nowDis);
				int gap = disLand[lid][cur] + lambda - V[cur].lb;
				curGap[cur] = max(0, gap);
			}
			if (node_continue[cur] && d_l[cur] >= nowDis)
			{
				{
					node_vis[u] = 1;
					if (!node_continue[u])
						vector<int>().swap(Dij_dis[u]);
					return false;
				}
			}
		}
		else
		{

			for (int i = 0; i < Dij_size; i++)
			{
				if (source == i || (Muti_Dijsktra_vis[i][cur] != stamp_vis && !node_continue[nbr[i]] && left != right))
					continue;
				int u_index = binarySearch(group_d[group_index], (group_len[group_index] + Dij_dis[right][cur] - Dij_dis[left][cur]) / 2, group_len[group_index], Dij_dis[nbr[0]][cur], Dij_dis[nbr[1]][cur]);
				int d = min(Dij_dis[left][cur] + group_d[group_index][u_index], Dij_dis[right][cur] + group_len[group_index] - group_d[group_index][u_index]);
				if (Muti_Dijsktra_vis[0][cur] == stamp_vis && Muti_Dijsktra_vis[1][cur] == stamp_vis && curGap[cur] != 0)
				{
					update_lb(cur, d);
					int gap = disLand[lid][cur] + lambda - V[cur].lb;
					curGap[cur] = max(0, gap);
				}
				if (Muti_Dijsktra_vis[0][cur] == stamp_vis && Muti_Dijsktra_vis[1][cur] == stamp_vis)
				{
					if (node_continue[cur] && (d_l[cur] >= Dij_dis[nbr[0]][cur] || d_l[cur] >= Dij_dis[nbr[1]][cur]))
					{
						if (d_l[cur] >= d)
						{
							for (int i = 0; i < group_mem[group_index].size(); i++)
							{
								node_vis[group_mem[group_index][i]] = 1;
							}
							if (!node_continue[left])
								vector<int>().swap(Dij_dis[left]);
							if (!node_continue[right])
								vector<int>().swap(Dij_dis[right]);
							return false;
						}
					}
					// if (Muti_Dijsktra_vis[0][cur] == stamp_vis && Muti_Dijsktra_vis[1][cur] == stamp_vis && curGap[cur] != 0)
					// {
					// 	update_lb(cur, d);
					// 	int gap = disLand[lid][cur] + lambda - V[cur].lb;
					// 	curGap[cur] = max(0, gap);
					// }
				}
			}
		}
		update_lb(nbr[source], nowDis);
		if (V[cur].lb < V[cur].ub)
			d_l[nbr[source]] = min(d_l[nbr[source]], V[cur].lb - nowDis);
		for (int i = 0; i < deg[cur]; i++)
		{
			int d = nowDis + weight[cur][i];
			int v = con[cur][i];
			if ((Muti_Dijsktra_dis[source][v] != stamp_dis) || (Dij_dis[nbr[source]][v] > d))
			{
				Muti_Dijsktra_dis[source][v] = stamp_dis;
				Dij_dis[nbr[source]][v] = d;
				pq.push(MEM{v, source, d});
			}
		}
	}

	if (deg[u] == 2)
	{
		for (int i = 0; i < group_mem[group_index].size(); i++)
		{
			node_vis[group_mem[group_index][i]] = 1;
		}
		V[left].ub = V[left].lb;
		V[right].ub = V[right].lb;
		node_continue[left] = node_continue[right] = 1;
	}
	else
	{
		node_vis[u] = 1;
		node_continue[u] = 1;
		V[u].ub = V[u].lb;
		curGap[u] = 0;
	}
	return true;
}

bool CoreTree::Dijkstra_Judge1(int u, vector<int> &node_vis, vector<int> &node_continue)
{
	priority_queue<MEM> pq;
	vector<int> nbr;
	int Dij_size = 0;
	bool flag_chain;
	stamp_vis++;
	stamp_dis++;
	int group_index = node_g[u];
	int left = group_l[group_index], right = group_r[group_index];
	if (deg[u] == 2)
	{
		if (node_continue[left] == 1 && node_continue[right] == 1)
			return true;
		if (!node_continue[left])
		{
			pq.push(MEM{left, Dij_size, 0});
			Dij_dis[left].resize(n);
			Dij_dis[left][left] = 0;
			Muti_Dijsktra_dis[Dij_size][left] = stamp_dis;
			nbr.push_back(left);
			Dij_size++;
		}
		if (!node_continue[right] && left != right)
		{
			pq.push(MEM{right, Dij_size, 0});
			Dij_dis[right].resize(n);
			Dij_dis[right][right] = 0;
			Muti_Dijsktra_dis[Dij_size][right] = stamp_dis;
			nbr.push_back(right);
			Dij_size++;
		}
		flag_chain = true;
		left = group_mem[group_index].front(), right = group_mem[group_index].back();
	}
	else
	{
		pq.push(MEM{u, 0, 0});
		Dij_dis[u].resize(n);
		Dij_dis[u][u] = 0;
		Muti_Dijsktra_dis[Dij_size][u] = stamp_dis;
		nbr.push_back(u);
		flag_chain = false;
		Dij_size++;
	}

	while (!pq.empty())
	{
		nnnn++;
		MEM member = pq.top();
		int cur = member.cur;
		int source = member.source;
		int nowDis = member.dis;
		pq.pop();

		if (Muti_Dijsktra_vis[source][cur] == stamp_vis)
		{
			continue;
		}
		Muti_Dijsktra_vis[source][cur] = stamp_vis;
		if (flag_chain == false)
		{
			if (node_continue[cur] && d_l[cur] >= nowDis)
			{
				{
					node_vis[u] = 1;
					if (!node_continue[u])
						vector<int>().swap(Dij_dis[u]);
					return false;
				}
			}
		}
		else
		{
			for (int i = 0; i < Dij_size; i++)
			{
				if (source == i || (Muti_Dijsktra_vis[i][cur] != stamp_vis && !node_continue[nbr[i]] && left != right))
					continue;
				if (node_continue[cur] && (d_l[cur] >= Dij_dis[nbr[0]][cur] || d_l[cur] >= Dij_dis[nbr[1]][cur]))
				{
					int u_index = binarySearch(group_d[group_index], (group_len[group_index] + Dij_dis[right][cur] - Dij_dis[left][cur]) / 2, group_len[group_index], Dij_dis[nbr[0]][cur], Dij_dis[nbr[1]][cur]);
					int d = min(Dij_dis[left][cur] + group_d[group_index][u_index], Dij_dis[right][cur] + group_len[group_index] - group_d[group_index][u_index]);
					if (d_l[cur] >= d)
					{
						for (int i = 0; i < group_mem[group_index].size(); i++)
						{
							node_vis[group_mem[group_index][i]] = 1;
						}
						if (!node_continue[left])
							vector<int>().swap(Dij_dis[left]);
						if (!node_continue[right])
							vector<int>().swap(Dij_dis[right]);
						return false;
					}
				}
			}
		}

		for (int i = 0; i < deg[cur]; i++)
		{
			int d = nowDis + weight[cur][i];
			int v = con[cur][i];
			if ((Muti_Dijsktra_dis[source][v] != stamp_dis) || (Dij_dis[nbr[source]][v] > d))
			{
				Muti_Dijsktra_dis[source][v] = stamp_dis;
				Dij_dis[nbr[source]][v] = d;
				pq.push(MEM{v, source, d});
			}
		}
	}

	if (deg[u] == 2)
	{
		for (int i = 0; i < group_mem[group_index].size(); i++)
		{
			node_vis[group_mem[group_index][i]] = 1;
		}
		node_continue[left] = node_continue[right] = 1;
	}
	else
	{
		node_vis[u] = 1;
		node_continue[u] = 1;
	}
	return true;
}

void CoreTree::Local_Dijkstra(int u, int stamp)
{
	priority_queue<pii, vector<pii>, greater<pii>> pq;

	int nbr_size = tree[tree[u].rid].nbr.size();
	pq.push({0, u});
	Dijsktra_dis[u] = 0;
	stamp_dis++;
	stamp_vis++;
	stamp_d[u] = stamp_dis;

	while (!pq.empty())
	{
		int curDist = pq.top().first;
		int cur = pq.top().second;
		pq.pop();

		if (stamp_v[cur] == stamp_vis)
			continue;
		stamp_v[cur] = stamp_vis;
		if (vi[cur] == stamp && nbr_size != 1)
		{
			for (int i = 0; i < nbr_size; i++)
			{
				int v = tree[tree[u].rid].nbr[i];
				if ((stamp_d[v] != stamp_dis) || (Dijsktra_dis[v] > Dijsktra_dis[cur] + update_dis[id_to_index[cur]][tree[tree[u].rid].nbr[i]]))
				{
					Dijsktra_dis[v] = Dijsktra_dis[cur] + update_dis[id_to_index[cur]][tree[tree[u].rid].nbr[i]];
					stamp_d[v] = stamp_dis;
					pq.push({Dijsktra_dis[v], v});
				}
			}
		}
		for (int i = 0; i < deg[cur]; i++)
		{
			int v = con[cur][i];
			int d = weight[cur][i];
			if ((stamp_d[v] != stamp_dis || Dijsktra_dis[v] > Dijsktra_dis[cur] + d) && (rank[v] != -1 || vi[v] == stamp))
			{
				Dijsktra_dis[v] = Dijsktra_dis[cur] + d;
				stamp_d[v] = stamp_dis;
				pq.push({Dijsktra_dis[v], v});
			}
		}
	}
}
void CoreTree::Local_Dijkstra(int u, int stamp, vector<vector<int>> &up_dis)
{

	int nbr_size = tree[tree[u].rid].nbr.size();
	int m = 0;
	priority_queue<pii, vector<pii>, greater<pii>> pq;
	int cnt = 0;
	pq.push({0, u});
	Dijsktra_dis[u] = 0;
	stamp_dis++;
	stamp_vis++;
	stamp_d[u] = stamp_dis;

	while (!pq.empty())
	{

		int curDist = pq.top().first;
		int cur = pq.top().second;
		pq.pop();
		m++;

		if (stamp_v[cur] == stamp_vis)
			continue;
		stamp_v[cur] = stamp_vis;
		if (vi[cur] == stamp && nbr_size != 1)
		{
			for (int i = 0; i < nbr_size; i++)
			{
				int v = tree[tree[u].rid].nbr[i];
				if ((stamp_d[v] != stamp_dis) || (Dijsktra_dis[v] > Dijsktra_dis[cur] + up_dis[id_to_index[cur]][i]))
				{
					Dijsktra_dis[v] = Dijsktra_dis[cur] + up_dis[id_to_index[cur]][i];
					stamp_d[v] = stamp_dis;
					pq.push({Dijsktra_dis[v], v});
				}
			}
		}
		for (int i = 0; i < deg[cur]; i++)
		{
			int v = con[cur][i];
			int d = weight[cur][i];
			if ((stamp_d[v] != stamp_dis || Dijsktra_dis[v] > Dijsktra_dis[cur] + d) && (rank[v] != -1 || vi[v] == stamp))
			{
				Dijsktra_dis[v] = Dijsktra_dis[cur] + d;
				stamp_d[v] = stamp_dis;
				pq.push({Dijsktra_dis[v], v});
			}
		}
	}
}

void CoreTree::Local_Dij(int u, int lambda, int lid, bool flag)
{
	queue<int> toVisit;
	int nbr_size = tree[tree[u].rid].nbr.size();
	int u_index = root_to_idex[tree[u].rid];
	int fa_index = father[u_index];
	stamp = ++stamp_dis;
	for (int i = 0; i < tree[tree[u].rid].nbr.size(); i++)
	{
		vi[tree[tree[u].rid].nbr[i]] = stamp;
		id_to_index[tree[tree[u].rid].nbr[i]] = i;
	}
	Local_Dijkstra(u, stamp, sum_update_dis[father[root_to_idex[tree[u].rid]]]);

	for (int i = 0; i < root_dis[fa_index].size(); i++)
	{
		int v = root_dis[fa_index][i].id;
		int dis = Dijsktra_dis[v];

		int id = v;
		int lb = V[id].lb, ub = V[id].ub;
		if (lb < ub)
		{
			update_lb(id, dis);
			// if (flag)
			// {
			int gap = disLand[lid][id] + lambda - V[id].lb;
			curGap[id] = max(0, gap);
			// }
		}
	}
	if (flag)
	{
		while (!H.empty())
		{
			auto [g, id] = H.front();
			if (curGap[id] == 0 || disLand[lid][id] + lambda - V[id].lb <= 0)
			{
				H.pop();
			}
			else
				break;
		}
	}
}

bool CoreTree::check(int u, vector<int> &visited, bool flag)
{
	if (rank[u] == -1)
	{
		return false;
	}
	int u_index = root_to_idex[tree[u].rid];
	int fa_index = father[u_index];

	if (!visited[fa_index])
	{
		Compute_root_dis(u);
		visited[fa_index] = 1;
	}
	if (candidate[u])
	{
		return false;
	}
	else
		return true;
}
void CoreTree::Muti_Dijsktra(int u)
{
	priority_queue<MEM> pq;
	vector<int> nbr;
	vector<int> sum_result;
	vector<vector<int>> nbr_vis;

	int nbr_size = tree[tree[u].rid].nbr.size();

	if (nbr_size == 1)
		return;
	int sum = 0;
	nbr.assign(nbr_size, 0);
	sum_result.resize(nbr_size);
	nbr_vis.resize(nbr_size);
	stamp_vis++;
	stamp_dis++;

	for (int i = 0; i < nbr_size; i++)
	{
		nbr[i] = tree[tree[u].rid].nbr[i];
		nbr_vis[i].assign(nbr_size, 0);
	}
	for (int i = 0; i < nbr_size; i++)
	{
		update_dis[i][nbr[i]] = 0;
		Muti_Dijsktra_dis[i][nbr[i]] = stamp_dis;
		pq.push(MEM{nbr[i], i, 0});
	}
	if (pq.empty())
		return;

	while (!pq.empty())
	{
		MEM member = pq.top();
		int cur = member.cur;
		int source = member.source;
		int nowDis = member.dis;
		pq.pop();

		if (Muti_Dijsktra_vis[source][cur] == stamp_vis)
		{
			continue;
		}
		Muti_Dijsktra_vis[source][cur] = stamp_vis;
		for (int i = 0; i < nbr_size; i++)
		{
			if (source == i || Muti_Dijsktra_vis[i][cur] != stamp_vis || nbr_vis[i][source] || nbr_vis[source][i] || sum_result[i] == nbr_size - 1 || update_dis[source][nbr[i]] < nowDis + update_dis[i][cur])
				continue;
			if (update_dis[source][nbr[i]] < nowDis + update_dis[i][cur])
			{
				update_dis[source][nbr[i]] = nowDis + update_dis[i][cur];
				update_dis[i][nbr[source]] = nowDis + update_dis[i][cur];
			}
			if (update_dis[source][nbr[i]] <= nowDis)
			{
				sum_result[source]++;
				sum_result[i]++;
				nbr_vis[i][source] = nbr_vis[source][i] = 1;
			}
		}
		for (int i = 0; i < deg[cur]; i++)
		{
			int d = nowDis + weight[cur][i];
			int v = con[cur][i];
			if ((Muti_Dijsktra_dis[source][v] != stamp_dis) || (update_dis[source][v] > d))
			{
				Muti_Dijsktra_dis[source][v] = stamp_dis;
				update_dis[source][v] = d;
				pq.push(MEM{v, source, d});
			}
		}

		sum = 0;
		for (int i = 0; i < nbr_size; i++)
		{
			sum += sum_result[i];
		}
		if (sum == nbr_size * (nbr_size - 1))
		{
			break;
		}
	}
}

void CoreTree::Compute_root_dis(int u)
{

	queue<int> que;
	DIS tmp;
	int v, d, father_index, cnt;

	father_index = father[root_to_idex[tree[u].rid]];
	for (int i = 0; i < root.size(); i++)
	{
		if (father[i] == father_index)
		{
			que.push(root[i]);
		}
	}
	Muti_Dijsktra(u);

	stamp = ++stamp_dis;
	for (int i = 0; i < tree[tree[u].rid].nbr.size(); i++)
	{
		vi[tree[tree[u].rid].nbr[i]] = stamp;
		id_to_index[tree[tree[u].rid].nbr[i]] = i;
	}
	vector<vector<int>> tmp_dis;
	int nbr_size = tree[tree[u].rid].nbr.size();
	tmp_dis.resize(nbr_size);
	for (int i = 0; i < nbr_size; i++)
	{
		for (int j = 0; j < nbr_size; j++)
			tmp_dis[i].push_back(update_dis[i][tree[tree[u].rid].nbr[j]]);
	}
	sum_update_dis[father[root_to_idex[tree[u].rid]]] = tmp_dis;
	while (!que.empty())
	{
		int cur = que.front();
		que.pop();
		tmp.clear();
		Local_Dijkstra(cur, stamp);
		for (int j = 0; j < tree[tree[cur].rid].nbr.size(); j++)
		{
			v = tree[tree[cur].rid].nbr[j];
			d = Dijsktra_dis[v];
			tmp.dis.push_back(d);
		}
		tmp.id = cur;
		root_dis[father_index].push_back(tmp);
		for (int j = 0; j < tree[cur].ch.size(); j++)
		{
			v = tree[cur].ch[j];
			que.push(v);
		}
	}

	vector<bool> used;
	used.assign(root_dis[father_index].size(), 1);

	sort(root_dis[father_index].begin(), root_dis[father_index].end(), [](const DIS &lhs, const DIS &rhs)
		 { return lhs.dis > rhs.dis; });

	for (int i = 0; i < root_dis[father_index].size(); i++)
	{
		for (int j = i + 1; j < root_dis[father_index].size(); j++)
		{
			cnt = 0;
			if (used[j] == 0)
				continue;
			for (int k = 0; k < tree[root[father_index]].nbr.size(); k++)
			{
				if (root_dis[father_index][i].dis[k] < root_dis[father_index][j].dis[k])
					break;
				if (root_dis[father_index][i].dis[k] >= root_dis[father_index][j].dis[k])
					cnt++;
			}
			if (cnt == tree[root[father_index]].nbr.size())
				used[j] = 0;
		}
	}
	for (int i = 0; i < root_dis[father_index].size(); i++)
	{
		if (used[i] == 1)
		{
			candidate[root_dis[father_index][i].id] = 1;
		}
	}
}

void CoreTree::Merge_root_dis()
{
	father.resize(root_n);
	int p = 0, cnt = 0, nn = 0, flag;
	vector<CM> f;
	f.resize(root_n);
	for (int i = 0; i < root.size(); i++)
	{
		f[i].id = i;
		f[i].ve = tree[root[i]].nbr;
		f[i].ve.push_back(root[i]);
		sort(f[i].ve.begin(), f[i].ve.end());
	}

	sort(f.begin(), f.end(), [](const CM &a, const CM &b)
		 {
			if (a.ve.size() == b.ve.size()) {
				for (int i = 0; i < a.ve.size(); i++) {
					if (a.ve[i] == b.ve[i]) continue;
					return a.ve[i] > b.ve[i];
				}
			}
			return a.ve.size() > b.ve.size(); });

	if (f.size() != 0)
		father[f[0].id] = f[0].id;

	for (int i = 1; i < f.size(); i++)
	{
		flag = 0;
		if (f[p].ve.size() != f[i].ve.size())
		{
			father[f[i].id] = f[i].id;
			p = i;
		}
		else
		{
			for (int j = 0; j < f[p].ve.size(); j++)
			{
				if (f[p].ve[j] != f[i].ve[j])
				{
					flag = 1;
					break;
				}
			}
			if (flag == 0)
			{
				father[f[i].id] = f[p].id;
				nn++;
			}
			else
			{
				father[f[i].id] = f[i].id;
				p = i;
			}
		}
	}
}
void CoreTree::Compute_group()
{
	vector<int> vis;
	int count = 0;
	vis.resize(n, 0);
	for (int i = 0; i < n; i++)
	{
		if (deg[i] == 1 || deg[i] > 2)
		{
			bool tmp_empty = false;
			node_g[i] = group_l.size();
			for (int j = 0; j < deg[i]; j++)
			{
				int right = con[i][j];
				if (vis[right] || deg[right] > 2 || deg[right] == 1)
					continue;
				vector<int> tmp = {i};
				vector<int> dd = {0};
				int d = 0, curr = right, prev = i;
				bool is_end = false;
				tmp_empty = true;
				d += weight[i][j];
				node_g[right] = group_l.size();
				vis[right] = 1;
				while (!is_end)
				{
					tmp.push_back(curr);
					dd.push_back(d);
					if (deg[curr] != 2)
					{
						is_end = true;
						group_l.push_back(i);
						group_r.push_back(curr);
						group_len.push_back(d);
						group_mem.push_back(tmp);
						group_d[count] = dd;
						count++;
					}
					else
					{
						for (int k = 0; k < deg[curr]; k++)
						{
							int next = con[curr][k];
							if (next != prev)
							{
								d += weight[curr][k];
								prev = curr;
								curr = next;
								right = curr;
								node_g[right] = count;
								if (deg[right] == 2)
									vis[right] = 1;
								break;
							}
						}
					}
				}
			}
			if (!tmp_empty)
			{
				vector<int> tmp = {i};
				vector<int> dd = {0};
				group_l.push_back(i);
				group_r.push_back(i);
				group_len.push_back(1);
				group_d[count] = dd;
				group_mem.push_back(tmp);
				count++;
			}
		}
	}
	group_size.resize(group_l.size());
	for (int i = 0; i < group_l.size(); i++)
	{
		group_size[i] = group_mem[i].size();
	}
}

void CoreTree::save_tmp_graph(int max_w, string bin_path)
{
	// printf("Saving Tmp Graph...\n");

	char stw[16];
	sprintf(stw, "%d", max_w);

	FILE *fout = fopen((bin_path + "tmp-" + string(stw) + ".bin").c_str(), "wb");
	fwrite(&n, sizeof(int), 1, fout);
	fwrite(&m_core, sizeof(long long), 1, fout);
	fwrite(rank.data(), sizeof(int), n, fout);
	vector<int> cored(n, 0);
	for (int i = 0; i < n; ++i)
		if (rank[i] == -1)
			cored[i] = (int)E[i].size();
	fwrite(cored.data(), sizeof(int), n, fout);
	for (int i = 0; i < n; ++i)
		if (rank[i] == -1)
			fwrite(E[i].data(), sizeof(pair<int, int>), E[i].size(), fout);
	fclose(fout);
	// printf("Tmp Graph Saved!\n");
}
void CoreTree::load_tmp_graph(int max_w, string bin_path)
{
	// printf("Loading Tmp Graph...\n");

	char stw[16];
	sprintf(stw, "%d", max_w);

	FILE *fin = fopen((bin_path + "tmp-" + string(stw) + ".bin").c_str(), "rb");
	fread(&n, sizeof(int), 1, fin);
	n_core = 0;
	fread(&m_core, sizeof(long long), 1, fin);
	rank.resize(n);
	fread(rank.data(), sizeof(int), n, fin);

	int n_bc = 0;
	core_deg = new int[n];
	core_dat = new pair<int, int>[m_core];
	core_con = new pair<int, int> *[n];
	fread(core_deg, sizeof(int), n, fin);
	fread(core_dat, sizeof(pair<int, int>), m_core, fin);
	fclose(fin);
	// printf("Tmp Graph Loaded!\n");
}

void CoreTree::save_label_tree(int max_w, string bin_path)
{
	// printf("Saving Tree Label...\n");
	char stw[16];
	sprintf(stw, "%d", max_w);

	FILE *fout = fopen((bin_path + "label-tree-" + string(stw) + ".bin").c_str(), "wb");
	fwrite(&n, sizeof(int), 1, fout);
	fwrite(rank.data(), sizeof(int), n, fout);

	fwrite(&root_n, sizeof(int), 1, fout);
	fwrite(root.data(), sizeof(int), root_n, fout);

	root_to_idex.resize(n, -1);
	for (int i = 0; i < root.size(); i++)
		root_to_idex[root[i]] = i;
	fwrite(root_to_idex.data(), sizeof(int), n, fout);

	for (int i = 0; i < n; ++i)
		if (rank[i] >= 0)
		{
			TreeNode &tn = tree[i];
			fwrite(&tn.rid, sizeof(int), 1, fout);
			fwrite(&tn.rsize, sizeof(int), 1, fout);
			fwrite(&tn.h, sizeof(int), 1, fout);
			fwrite(&tn.w, sizeof(int), 1, fout);
			fwrite(&tn.ch_n, sizeof(int), 1, fout);
			fwrite(tn.nbr.data(), sizeof(int), tn.w, fout);
			int ch_size = tn.ch.size();
			fwrite(tn.ch.data(), sizeof(int), tn.ch_n, fout);
		}
	fclose(fout);

	// printf("Tree Label Saved!\n");
}

void CoreTree::load_label_tree(int max_w, string bin_path)
{
	// printf("Loading Tree Label...\n");
	char stw[16];
	sprintf(stw, "%d", max_w);

	FILE *fin = fopen((bin_path + "label-tree-" + string(stw) + ".bin").c_str(), "rb");

	fread(&n, sizeof(int), 1, fin);
	rank.resize(n);

	fread(rank.data(), sizeof(int), n, fin);
	n_core = 0;
	for (int i = 0; i < n; ++i)
		if (rank[i] == -1)
			++n_core;
	tree.resize(n);

	fread(&root_n, sizeof(int), 1, fin);
	root.resize(root_n);
	fread(root.data(), sizeof(int), root_n, fin);
	root_to_idex.resize(n);
	fread(root_to_idex.data(), sizeof(int), n, fin);
	int pos = 0;
	for (int i = 0; i < n; ++i)
		if (rank[i] >= 0)
		{
			TreeNode &tn = tree[i];
			fread(&tn.rid, sizeof(int), 1, fin);
			fread(&tn.rsize, sizeof(int), 1, fin);
			fread(&tn.h, sizeof(int), 1, fin);
			fread(&tn.w, sizeof(int), 1, fin);
			fread(&tn.ch_n, sizeof(int), 1, fin);
			tn.nbr.resize(tn.w);
			tn.anc.resize(tn.h);
			tn.dis.resize(tn.h);
			tn.ch.resize(tn.ch_n);
			fread(tn.nbr.data(), sizeof(int), tn.w, fin);
			fread(tn.ch.data(), sizeof(int), tn.ch_n, fin);
		}
	fclose(fin);
}

void CoreTree::load_label(int max_w, string bin_path, int numLand)
{
	this->max_w = max_w;
	this->numLand = numLand;
	load_label_tree(max_w, bin_path);
	load_tmp_graph(max_w, bin_path);
	init_query();
	load_graph(bin_path);
}

void CoreTree::decompose_tree(int max_w, int n_threads, string bin_path)
{
	if (con == NULL)
		load_graph(bin_path);
	double t1 = get_time();
	reduce(max_w, n_threads);
	create_tree();
	printf("Computing tree time:%f sec\n", get_time() - t1);
	save_label_tree(max_w, bin_path);
}

void CoreTree::init()
{
	stamp_dis = 2;
	stamp_vis = 2;
	V.resize(n);
	root_dis.resize(root_n);
	candidate.resize(n, 0);
	sum_update_dis.resize(root.size());
	vi.resize(n);
	id_to_index.resize(n);
	stamp_d.assign(n, 0);
	stamp_v.assign(n, 0);
	Dijsktra_dis.assign(n, 0);
	Muti_Dijsktra_dis.resize(max(2, max_w));
	Muti_Dijsktra_vis.resize(max(2, max_w));
	update_dis.resize(max(2, max_w));
	ListToIndex.resize(numLand);
	d_l.resize(n, INT_MAX);
	node_g.resize(n, -1);
	group_d.resize(n);
	Dij_dis.resize(n);
	curGap.resize(n, INT_MAX);
	for (int i = 0; i < max(2, max_w); i++)
	{
		update_dis[i].assign(n, INT_MAX);
		Muti_Dijsktra_vis[i].assign(n, 0);
		Muti_Dijsktra_dis[i].assign(n, 0);
	}
	for (int i = 0; i < n; ++i)
	{
		sort(V[i].adj.begin(), V[i].adj.end());
		V[i].len = (int)V[i].adj.size();
		V[i].ub = POSINF;
		V[i].lb = NEGINF;
	}
}

void CoreTree::compute_tree_label(int x, int rsize, vector<TreeNode *> &s)
{
	s.push_back(&tree[x]);
	TreeNode &tn = tree[x];
	tn.dis.resize(tn.h);
	int pos = 0;
	vector<int> p(tn.w);
	for (int j = 0; j < tn.w; ++j)
	{
		while (s[pos]->id != tn.nbr[j])
			++pos;
		p[j] = pos;
	}

	for (int i = 0; i < tn.h - 1; ++i)
	{
		tn.dis[i] = -1;
		for (int j = 0; j < tn.w; ++j)
		{
			int w = tn.cost[j], k = p[j], nowdis = -1;
			if (k <= i)
			{
				if (i >= rsize)
					nowdis = s[i]->dis[k];
				else if (k == i)
					nowdis = 0;
			}
			else if (k >= rsize)
				nowdis = s[k]->dis[i];
			if (nowdis >= 0 && (tn.dis[i] == -1 || nowdis + w < tn.dis[i]))
				tn.dis[i] = min(nowdis + w, MAXD);
		}
	}
	tn.dis[tn.h - 1] = 0;
	for (int &u : tree[x].ch)
		compute_tree_label(u, rsize, s);
	s.pop_back();
}

void CoreTree::compute_tree_label()
{
	printf("Computing Tree Label...\n");
	vector<TreeNode *> s;
	for (int v = 0; v < n; ++v)
		if (rank[v] >= 0 && tree[v].f == -1)
		{
			s.clear();
			for (int i = 0; i < tree[v].w; ++i)
				s.push_back(&tree[tree[v].nbr[i]]);
			compute_tree_label(v, tree[v].rsize, s);
		}
	double t_size = 0;
	int maxdis = 0;
	for (int v = 0; v < n; ++v)
	{
		if (rank[v] >= 0)
		{
			t_size += tree[v].dis.size() * 1.0 * (sizeof(int) + sizeof(dint));
			vector<pair<int, int>>().swap(E[v]);
			for (auto &d : tree[v].dis)
				maxdis = max(maxdis, (int)d);
		}
		else
			vector<pair<int, int>>(E[v]).swap(E[v]);
	}

	printf("Tree Label Computed, t=%0.3lf secs, maxdis=%d, tree label size=%0.3lf MB\n", omp_get_wtime() - t, maxdis, t_size / (1024.0 * 1024.0));
}

void CoreTree::create_tree()
{
	// printf("Creating Tree...\n");
	tree.resize(n);
	for (int u = 0; u < n; ++u)
		tree[u].id = u;
	vector<pair<int, int>> v_pair;
	int maxh = 0, cnt_root = 0, maxdep = 0, max_sub_tree = 1;
	vector<int> tcnt(n, 0);
	double tw = 0;
	for (int i = (int)ord.size() - 1; i >= 0; --i)
	{
		int x = ord[i];

		TreeNode &tn = tree[x];
		v_pair.clear();
		for (int j = 0; j < (int)nbr[x].size(); ++j)
		{
			int y = nbr[x][j];
			if (rank[y] == -1)
				v_pair.push_back(make_pair(n, j));
			else
				v_pair.push_back(make_pair(rank[y], j));
		}
		sort(v_pair.begin(), v_pair.end());
		reverse(v_pair.begin(), v_pair.end());
		int w = (int)nbr[x].size();
		tn.nbr.resize(w);
		// tn.cost.resize(w);
		for (int j = 0; j < w; ++j)
		{
			tn.nbr[j] = nbr[x][v_pair[j].second];
			// tn.cost[j] = cost[x][v_pair[j].second];
		}

		tn.w = w;
		tn.id = x;
		tn.f = -1;
		tn.ch_n = 0;
		for (auto &u : nbr[x])
			if (rank[u] != -1 && (tn.f == -1 || rank[u] < rank[tn.f]))
				tn.f = u;
		if (tn.f == -1)
		{
			tn.h = tn.w + 1;
			++cnt_root;
			++tcnt[x];
			tn.rid = x;
			tn.rsize = tn.w;
			tn.anc.push_back(x);
			root.push_back(x);
		}
		else
		{
			tn.h = tree[tn.f].h + 1;
			tree[tn.f].ch.push_back(x);
			tree[tn.f].ch_n++;
			tn.rid = tree[tn.f].rid;
			++tcnt[tn.rid];
			max_sub_tree = max(max_sub_tree, tcnt[tn.rid]);
			tn.rsize = tree[tn.f].rsize;
			tn.anc = tree[tn.f].anc;
			tn.anc.push_back(x);
		}
		tw += tn.rsize;
		maxh = max(maxh, tn.h);
		maxdep = max(maxdep, (int)tn.anc.size());
	}
	root_n = cnt_root;

	// printf("Core tree constructed, maxh=%d, maxdep=%d, cnt_root=%d, max_stree=%d, avg_rsize=%0.3lf, t=%0.3lf secs\n",
	// 	maxh, maxdep, cnt_root, max_sub_tree, tw / (n - n_core), omp_get_wtime() - t);
}

void CoreTree::reduce(int max_w, int n_threads)
{
	omp_set_num_threads(n_threads);
	this->max_w = max_w;
	score.resize(n);
	_score.resize(n);
	vector<bool> changed(n, false);
	set<IntVal> q;
	nbr.resize(n);
	cost.resize(n);
	rank.resize(n);
	fill(rank.begin(), rank.end(), -1);
	int r = 0;

	for (int i = 0; i < n; ++i)
		score[i] = _score[i] = deg[i];

	// printf("Initializing q...");
	vector<bool> active(n, false);
	int riginal_core_number = 0;
	int cntt = 0;
	for (int u = 0; u < n; ++u)
	{
		if (deg[u] < max_w)
		{
			q.insert(IntVal(u));
		}
		active[u] = true;
	}

	E.resize(n);

	for (int u = 0; u < n; ++u)
		for (int i = 0; i < deg[u]; ++i)
		{
			E[u].push_back(make_pair(con[u][i], weight[u][i]));
		}

	int cnt = 0;
	vector<pair<int, int>> tmp;
	while (!q.empty())
	{
		int x = q.begin()->x;
		while (changed[x])
		{
			q.erase(x);
			score[x] = _score[x];
			q.insert(x);
			changed[x] = false;
			x = q.begin()->x;
		}
		if (score[x] >= max_w)
			break;
		ord.push_back(x);
		q.erase(x);
		rank[x] = r++;
		for (auto &it : E[x])
		{
			nbr[x].push_back(it.first);
			cost[x].push_back(it.second);
		}

		for (auto &y : nbr[x])
		{
			if (E[y].size() >= max_w * 2)
			{
				active[y] = false;
				q.erase(y);
			}
			if (!active[y])
				continue;
			for (size_t i = 0; i < E[y].size(); ++i)
			{
				if (E[y][i].first == x)
				{
					E[y].erase(E[y].begin() + i);
					break;
				}
			}
		}

		for (size_t i = 0; i < nbr[x].size(); ++i)
		{
			int u = nbr[x][i];
			if (!active[u])
			{
				E[u].push_back(make_pair(x, -cost[x][i]));
				continue;
			}
			tmp.clear();
			size_t j = 0, k = 0;
			while (j < nbr[x].size() && k < E[u].size())
				if (j == i)
					++j;
				else if (nbr[x][j] < E[u][k].first)
				{
					tmp.push_back(make_pair(nbr[x][j], cost[x][i] + cost[x][j]));
					++j;
				}
				else if (nbr[x][j] > E[u][k].first)
				{
					tmp.push_back(E[u][k]);
					++k;
				}
				else
				{
					if (E[u][k].second < cost[x][i] + cost[x][j])
						tmp.push_back(E[u][k]);
					else
						tmp.push_back(make_pair(nbr[x][j], cost[x][i] + cost[x][j]));
					++j;
					++k;
				}
			for (; j < nbr[x].size(); ++j)
				if (j != i)
					tmp.push_back(make_pair(nbr[x][j], cost[x][i] + cost[x][j]));
			for (; k < E[u].size(); ++k)
				tmp.push_back(E[u][k]);
			E[u] = tmp;
			_score[u] = (int)E[u].size();
			q.erase(u);
			score[u] = _score[u];
			q.insert(u);
			changed[u] = false;
		}

		if ((++cnt) * score[x] > 1000000)
		{
			// printf("%d nodes reduced, score[x]=%d, remaining size=%0.3lf%% t=%0.3lf secs\n",
			// 	r, (n - r) * 100.0 / n, score[x], omp_get_wtime() - t);
			cnt = 0;
		}
	}
	n_core = 0;
	m_core = 0;
	for (int u = 0; u < n; ++u)
		if (rank[u] == -1)
		{
			++n_core;
			m_core += (int)E[u].size();
		}

	// printf("Reducing finished, t=%0.3lf secs\nn_core=%d,m_core=%lld,node_rate=%0.3lf,edge_rate=%0.3lf\n",
	// 	omp_get_wtime() - t, n_core, m_core, n_core * 1.0 / n, m_core * 1.0 / m);
}

void CoreTree::load_graph(string bin_path)
{
	// printf("loading graph: %s\n", bin_path.c_str());
	long long p = 0;
	FILE *fin = fopen((bin_path + "graph-dis.bin").c_str(), "rb");
	fread(&n, sizeof(int), 1, fin);
	fread(&m, sizeof(long long), 1, fin);
	deg = new int[n];
	dat = new int[m];
	con = new int *[n];
	nid = new int[n];
	old = new int[n];
	new_old = new int[n];
	vector<int> weg;
	weg.resize(m);
	weight.resize(n);
	fread(deg, sizeof(int), n, fin);
	fread(dat, sizeof(int), m, fin);
	fread(nid, sizeof(int), n, fin);
	fread(old, sizeof(int), n, fin);
	fread(weg.data(), sizeof(int), m, fin);
	int cnt = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < deg[i]; j++)
		{
			weight[i].push_back(weg[cnt++]);
		}
	}

	fclose(fin);
	p = 0;
	for (int i = 0; i < n; ++i)
	{
		con[i] = dat + p;
		p += deg[i];
	}
	// printf("graph loaded, n_org=%d, n=%d, m=%lld\n", n_org, n, m);
}

bool CoreTree::get_edge(char *line, int &a, int &b, int &c, int num_cnt)
{
	if (!isdigit(line[0]))
		return false;
	vector<char *> v_num;
	int len = (int)strlen(line);
	for (int i = 0; i < len; ++i)
		if (!isdigit(line[i]) && line[i] != '.')
			line[i] = '\0';
		else if (i == 0 || !line[i - 1])
			v_num.push_back(line + i);
	if ((int)v_num.size() != num_cnt)
		return false;
	sscanf(v_num[0], "%d", &a);
	sscanf(v_num[1], "%d", &b);
	sscanf(v_num[2], "%d", &c);
	return true;
}

int CoreTree::get_num_cnt(string path)
{
	FILE *fin = fopen((path).c_str(), "r");
	char line[MAXLINE];
	int cnt = 0, min_cnt = 100;

	while (fgets(line, MAXLINE, fin) && cnt < 10)
	{
		if (!isdigit(line[0]))
			continue;
		vector<char *> v_num;
		int len = (int)strlen(line);
		for (int i = 0; i < len; ++i)
			if (!isdigit(line[i]) && line[i] != '.')
				line[i] = '\0';
			else if (i == 0 || !line[i - 1])
				v_num.push_back(line + i);
		if ((int)v_num.size() < 3)
			continue;
		min_cnt = min(min_cnt, (int)v_num.size());
		++cnt;
	}
	fclose(fin);
	return min_cnt;
}

void CoreTree::get_order(vector<pair<int, int>> *con, int n, int *o, int method)
{
	printf("method=%d\n", method);
	if (method == RANK_STATIC)
	{
		printf("Ranking Method = RANK_STATIC\n");
		// return val
		DV *f = new DV[n];
		for (int i = 0; i < n; ++i)
			f[i].id = i, f[i].val = con[i].size() * 1.0;
		sort(f, f + n);
		for (int i = 0; i < n; ++i)
			o[i] = f[i].id;
		delete[] f;
	}
}

void CoreTree::create_bin(string path, string graph_name, int rank_threads, int rank_method, int rank_max_minutes, int max_hops, bool merge_equv, string bin_path)
{
	string full_path = path + graph_name;
	FILE *fin = fopen((full_path).c_str(), "r");
	char line[MAXLINE];
	int n = 0, a, b, c, num_cnt = get_num_cnt(full_path);
	vector<pair<int, int>> el;
	vector<int> dl;
	long long cnt = 0, m = 0;

	printf("Loading text, num_cnt=%d...\n", num_cnt);
	while (fgets(line, MAXLINE, fin))
	{
		if (!get_edge(line, a, b, c, num_cnt))
			continue;
		if (a < 0 || b < 0 || a == b)
			continue;
		el.push_back(make_pair(a, b));
		dl.push_back(c);
		n = max(max(n, a + 1), b + 1);
		if ((++cnt) % (long long)10000000 == 0)
			printf("%lld lines finished\n", cnt);
	}
	fclose(fin);

	cout << "origional node:" << n << endl;
	vector<int> *con1 = new vector<int>[n];

	printf("Deduplicating...\n");

	for (size_t i = 0; i < el.size(); ++i)
	{
		con1[el[i].first].push_back(el[i].second);
		con1[el[i].second].push_back(el[i].first);
	}

	vector<int> degg;
	degg.resize(n);
	for (int i = 0; i < n; ++i)
	{
		if (con1[i].size() > 0)
		{
			sort(con1[i].begin(), con1[i].end());
			int p = 1;
			for (int j = 1; j < (int)con1[i].size(); ++j)
			{
				if (con1[i][j - 1] != con1[i][j])
				{
					con1[i][p++] = con1[i][j];
				}
			}

			con1[i].resize(p);
			m += p;
			degg[i] = p;
		}
	}

	int id = 0;
	vector<Val> t;
	t.clear();
	for (int i = 0; i < n; ++i)
	{
		t.push_back(Val(i, degg[i]));
	}
	sort(t.begin(), t.end());
	id = t[0].id;

	queue<int> to_visit;
	vector<int> vistied;
	vistied.resize(n, 0);
	vector<int> dis;
	dis.clear();
	dis.resize(n);

	vector<int> old_new;
	old_new.resize(n, -1);
	to_visit.push(id);
	vistied[id] = 1;
	cnt = 0;
	dis[id] = 0;
	old_new[id] = cnt++;

	while (!to_visit.empty())
	{
		int n_id = to_visit.front();
		to_visit.pop();

		for (vector<int>::iterator i = con1[n_id].begin();
			 i != con1[n_id].end();
			 i++)
		{
			if (vistied[*i] == 0)
			{
				vistied[*i] = 1;
				dis[*i] = dis[n_id] + 1;
				old_new[*i] = cnt++;
				to_visit.push(*i);
			}
		}
	}

	vector<pair<pair<int, int>, int>> n_el;
	cout << el.size() << endl;
	n = 0;
	m = 0;
	for (int i = 0; i < el.size(); i++)
	{
		int a = el[i].first;
		int b = el[i].second;
		int ww = dl[i];

		if (old_new[a] > -1 and old_new[b] > -1)
		{
			int n_a = old_new[a];
			int n_b = old_new[b];
			n_el.push_back(make_pair(make_pair(n_a, n_b), dl[i]));
			m++;
			n = max(max(n, n_a + 1), n_b + 1);
		}
	}

	printf("# conn nodes=%d,m= %d\n", n, m);
	m = 0;
	vector<pair<int, int>> *con = new vector<pair<int, int>>[n];
	vector<vector<int>> we;
	we.resize(n);

	for (long long i = 0; i < n_el.size(); ++i)
	{
		con[n_el[i].first.first].push_back(make_pair(n_el[i].first.second, n_el[i].second));
		con[n_el[i].first.second].push_back(make_pair(n_el[i].first.first, n_el[i].second));
	}
	for (int i = 0; i < n; ++i)
	{
		if (con[i].size() > 0)
		{
			sort(con[i].begin(), con[i].end());
			int p = 1;
			for (int j = 1; j < (int)con[i].size(); ++j)
				if (con[i][j - 1].first != con[i][j].first)
				{
					con[i][p++] = con[i][j];
				}
			con[i].resize(p);
			m += p;
			degg[i] = p;
		}
	}
	printf("n=%d %d\n", n, m);

	vector<pair<int, int>> *conn = new vector<pair<int, int>>[n];

	long long *f1 = new long long[n];
	memset(f1, 0, sizeof(long long) * n);

	long long *f2 = new long long[n];
	memset(f2, 0, sizeof(long long) * n);

	for (int i = 0; i < n; ++i)
		f1[i] = i, f2[i] = i;

	printf("Reordering...\n");
	int *f = new int[n];
	get_order(con, n, f, rank_method);

	int *old = new int[n], *nid = new int[n];
	for (int i = 0; i < n; ++i)
		nid[f[i]] = i;
	for (int i = 0; i < n; ++i)
		old[nid[i]] = i;

	for (int i = 0; i < n; ++i)
	{
		if (f1[i] != i)
			nid[i] = -nid[f1[i]] - 1;
		if (f2[i] != i)
			nid[i] = MAXN + nid[f2[i]];
	}
	cout << "n:" << n << ",m:" << m << endl;
	printf("Creating adjacency list...\n");
	int *dat = new int[m], *deg = new int[n], **adj = new int *[n];

	long long pos = 0;
	for (int i = 0; i < n; ++i)
	{
		adj[i] = dat + pos;
		pos += (int)con[old[i]].size();
	}
	memset(deg, 0, sizeof(int) * n);
	for (int i = 0; i < n; ++i)
	{
		int ii = old[i];
		for (int p = 0; p < (int)con[ii].size(); ++p)
		{
			int jj = con[ii][p].first;
			int j = nid[jj];
			adj[j][deg[j]++] = i;
		}
	}
	vector<vector<int>> weight;
	weight.resize(n);
	for (int i = 0; i < n; ++i)
	{
		int ii = old[i];
		for (int p = 0; p < (int)con[ii].size(); ++p)
		{
			int jj = con[ii][p].first;
			int j = nid[jj];
			weight[j].push_back(con[ii][p].second);
		}
	}
	vector<int> ww;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < deg[i]; j++)
		{
			ww.push_back(weight[i][j]);
		}
	}

	// printf("Saving binary...\n");

	FILE *fout = fopen((bin_path + "graph-dis.bin").c_str(), "wb");
	// cout << "n:" << n << endl;
	fwrite(&n, sizeof(int), 1, fout);
	fwrite(&m, sizeof(long long), 1, fout);
	fwrite(deg, sizeof(int), n, fout);
	fwrite(dat, sizeof(int), m, fout);
	fwrite(nid, sizeof(int), n, fout);
	fwrite(old, sizeof(int), n, fout);
	fwrite(ww.data(), sizeof(int), m, fout);
	fclose(fout);

	printf("Created binary file, n = %d, m = %lld\n", n, m);
}

void CoreTree::create_landmark()
{
	select_landmark();
	distance_landmark();
	update_bound();
	assign_group();
}
void CoreTree::select_landmark()
{
	landId.clear();

	vector<Val> l;
	for (int i = 0; i < n; ++i)
		l.push_back(Val(i, V[i].len));

	sort(l.begin(), l.end());

	int maxx = INT_MAX, bs, id;
	vector<int> di;

	for (int i = 0; i < numLand; ++i)
		landId.push_back(l[i].id);
}

void CoreTree::distance_landmark()
{
	disLand.resize(numLand, vector<int>(n, INT_MAX));
	for (int i = 0; i < numLand; ++i)
	{
		int uid = landId[i];
		Dijkstra(uid, disLand[i]);
	}
}

void CoreTree::update_bound()
{
	listLand.resize(numLand);
	// cout << V[1].lb << "," << V[1].ub << endl;
	for (int i = 0; i < numLand; ++i)
	{
		ListToIndex[i].assign(n, 0);
		for (int j = 0; j < n; ++j)
		{
			listLand[i].push_back(Val(j, disLand[i][j]));
			update_lb(j, disLand[i][j]);
		}
		sort(listLand[i].begin(), listLand[i].end());
		int ecc = listLand[i][0].val;

		for (int j = 0; j < n; ++j)
		{
			update_lb(j, ecc - disLand[i][j]);
			update_ub(j, ecc + disLand[i][j]);
			int gap = V[j].ub - V[j].lb;
			// if (j == 1)
			// 	cout << V[j].lb << "," << V[j].ub << endl;
			if (gap < curGap[j])
			{

				curGap[j] = gap;
				if (curGap[j] != 0)
					H.emplace(gap, j);
			}
		}
		for (int j = 0; j < n; j++)
		{
			ListToIndex[i][listLand[i][j].id] = j;
		}
	}
}

void CoreTree::assign_group()
{
	nodeGroup.resize(numLand);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < numLand; j++)
			nodeGroup[j].push_back(i);
	}
}

int CoreTree::get_unmch(int lid)
{
	int unmch = 0;
	for (int i = 0; i < nodeGroup[lid].size(); ++i)
	{
		int id = nodeGroup[lid][i];
		if (V[id].ub != V[id].lb)
		{
			unmch++;
		}
	}

	return unmch;
}
int CoreTree::get_unmch()
{
	int unmch = 0;
	for (int i = 0; i < n; ++i)
	{
		int id = i;
		if (V[id].ub != V[id].lb)
		{
			unmch++;
		}
	}
	return unmch;
}
void CoreTree::update_lb(int v, int lb)
{
	if (lb > V[v].lb)
		V[v].lb = lb;
}

void CoreTree::update_ub(int v, int ub)
{
	if (ub < V[v].ub)
		V[v].ub = ub;
}

int CoreTree::binarySearch(const vector<int> &arr, int target, int g_len, int d1, int d2)
{
	int u = -1;
	auto lower = upper_bound(arr.begin(), arr.end(), target);
	int maxx = 0;
	if (lower != arr.begin())
	{
		int diff = min(d1 + *(lower - 1), d2 + g_len - *(lower - 1));
		if (diff > maxx)
		{
			maxx = diff;
			u = (lower - 1) - arr.begin();
		}
	}
	if (lower != arr.end())
	{
		int diff = min(d1 + *lower, d2 + g_len - *lower);
		if (diff > maxx)
		{
			maxx = diff;
			u = lower - arr.begin();
		}
	}
	return u;
}
void CoreTree::txt_to_file(string graph_name)
{
	string file_name = "txt/" + graph_name;
	FILE *out = fopen(file_name.c_str(), "w");
	if (out == NULL)
	{
		fprintf(stderr, "can't write the graph file \n");
		exit(1);
	}

	vector<pair<int, int>> v;
	for (int i = 0; i < n; i++)
	{
		fprintf(out, "%d %d\n", i, V[i].lb);
	}

	fclose(out);

	// FILE* out1 = fopen("txt/BFS_result.txt", "w");
	// if (out1 == NULL)
	// {
	// 	fprintf(stderr, "can't write the graph file \n");
	// 	exit(1);
	// }

	// vector<int> diss;
	// int ecc = 0;
	// for (int i = 0; i < n; i++)
	// {
	// 	ecc = 0;
	// 	Dijkstra(i, diss);
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		ecc = max(ecc, diss[j]);
	// 	}
	// 	fprintf(out1, "%d %d\n", i, ecc);
	// }
	// fclose(out1);
}

void CoreTree::compute_ecc(bool flag)
{
	int lambcnt = 0, cnt = 0, lid = 0, max_d, id = -1, max_id = -1, max_lb = -1;

	vector<int> visited;
	vector<int> node_vis;
	vector<int> node_continue;
	vector<int> d;
	vector<bool> active;

	node_continue.resize(n, 0);
	node_vis.resize(n, 0);
	visited.resize(n);
	active.assign(numLand, false);
	d.resize(numLand, 0);
	if (flag)
	{
		while (true)
		{
			// cout << lambcnt << "," << V[1].lb << "," << V[1].ub << endl;
			compute_ecc(lid, lambcnt, cnt, visited, node_vis, node_continue, max_id, max_lb, flag);

			auto [g, id] = H.front();
			if (H.empty())
				break;
			lambcnt++;
		}
	}
	else
	{
		while (true)
		{
			compute_ecc1(lid, lambcnt, cnt, visited, node_vis, node_continue, max_id, max_lb, flag);
			if (max_id != -1 && disLand[lid][max_id] + listLand[lid][lambcnt].val <= max_lb)
			{
				break;
			}
			lambcnt++;
		}
	}
}

void CoreTree::compute_ecc(int lid, int &lambcnt, int &cnt, vector<int> &visited, vector<int> &node_vis, vector<int> &node_continue, int &max_id, int &max_lb, bool flag)
{
	int lambda = listLand[lid][lambcnt].val;
	int uid = listLand[lid][lambcnt].id;
	int group_index = node_g[uid];

	if (check(uid, visited, flag))
	{
		Local_Dij(uid, lambda, lid, flag);
		return;
	}
	if (Dijkstra_Judge(uid, lid, lambda, node_vis, node_continue))
	{
		cnt++;
		while (!H.empty())
		{
			auto [g, id] = H.front();
			if (curGap[id] == 0 || disLand[lid][id] + lambda - V[id].lb <= 0)
			{
				H.pop();
			}
			else
				break;
		}
	}
	else
	{

		if (lambcnt > 0)
		{
			int gap = listLand[lid][lambcnt - 1].val - listLand[lid][lambcnt].val;

			while (!H.empty())
			{
				auto [g, id] = H.front();
				if (curGap[id] - gap <= 0 || disLand[lid][id] + lambda - V[id].lb <= 0)
				{
					V[id].ub = V[id].lb;
					curGap[id] = max(0, curGap[id] - gap);
					H.pop();
				}
				else
				{
					curGap[id] = curGap[id] - gap;
					break;
				}
			}
		}
		return;
	}
}

void CoreTree::compute_ecc1(int lid, int &lambcnt, int &cnt, vector<int> &visited, vector<int> &node_vis, vector<int> &node_continue, int &max_id, int &max_lb, bool flag)
{
	int lambda = listLand[lid][lambcnt].val;
	int uid = listLand[lid][lambcnt].id;
	int group_index = node_g[uid];

	if (check(uid, visited, flag))
	{
		Local_Dij(uid, lambda, lid, flag);
		return;
	}
	if (Dijkstra_Judge1(uid, node_vis, node_continue))
	{
		cnt++;
		int ecc = NEGINF, max_difference = 0;
		int group_index = node_g[uid];
		int left = group_l[group_index], right = group_r[group_index];
		if (deg[uid] == 2)
		{
			for (int i = 0; i < nodeGroup[lid].size(); ++i)
			{
				int id = nodeGroup[lid][i];
				int lb = V[id].lb, ub = V[id].ub;
				int u, left_distance, right_distance;
				if (node_g[id] == group_index || id == left || id == right)
				{
					left_distance = Dij_dis[left][id];
					right_distance = Dij_dis[right][id];
				}
				else
				{
					u = binarySearch(group_d[group_index], (group_len[group_index] + Dij_dis[right][id] - Dij_dis[left][id]) / 2, group_len[group_index], Dij_dis[left][id], Dij_dis[right][id]);
					left_distance = Dij_dis[left][id] + group_d[group_index][u];
					right_distance = Dij_dis[right][id] + group_len[group_index] - group_d[group_index][u];
				}
				if (lb < ub)
				{
					int d = min(left_distance, right_distance);
					update_lb(id, d);
					// if (V[id].lb == V[id].ub)
					// {
					// 	swap(nodeGroup[lid][i], nodeGroup[lid].back());
					// 	nodeGroup[lid].pop_back();
					// 	i--;
					// 	continue;
					// }
					if (disLand[lid][id] + lambda - V[id].lb > max_difference)
					{
						max_difference = disLand[lid][id] + lambda - V[id].lb;
						max_id = id;
						max_lb = V[id].lb;
					}
					d_l[left] = min(d_l[left], V[id].lb - Dij_dis[left][id]);
					d_l[right] = min(d_l[right], V[id].lb - Dij_dis[right][id]);
				}
			}
		}
		else
		{

			for (int i = 0; i < n; i++)
			{
				ecc = max(ecc, Dij_dis[uid][i]);
			}
			V[uid].lb = V[uid].ub = ecc;
			for (int i = 0; i < nodeGroup[lid].size(); ++i)
			{
				int id = nodeGroup[lid][i];
				int lb = V[id].lb, ub = V[id].ub;
				if (lb < ub)
				{
					update_lb(id, Dij_dis[uid][id]);
					// if (V[id].lb == V[id].ub)
					// {
					// 	swap(nodeGroup[lid][i], nodeGroup[lid].back());
					// 	nodeGroup[lid].pop_back();
					// 	i--;
					// 	continue;
					// }
					if (disLand[lid][id] + lambda - V[id].lb > max_difference)
					{
						max_difference = disLand[lid][id] + lambda - V[id].lb;
						max_id = id;
						max_lb = V[id].lb;
					}
					d_l[uid] = min(d_l[uid], V[id].lb - Dij_dis[uid][id]);
				}
			}
		}
	}
	else
	{
		return;
	}
}

#endif /* CORETREELABELLING_H_ */
