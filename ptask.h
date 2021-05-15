#ifndef	PTHASK_H
#define	PTHASK_H

#include <time.h>

//	Time management functions
extern void time_copy (struct timespec *td, struct timespec ts);
extern void time_add_ms (struct timespec *t, int ms);
extern int time_cmp (struct timespec t1, struct timespec t2);

//	Task management functions
extern void ptask_init(int scheduler);
extern int task_create(void* (*task) (void*), int i, int period, int drel, int prio);
extern int get_task_index(void* arg);
extern int get_task_period(int i);
extern int get_task_dmiss(int i);
extern void set_activation(int i);
extern int deadline_miss(int i);
extern void wait_for_period (int i);
extern void wait_for_task_end(int i);
#endif


