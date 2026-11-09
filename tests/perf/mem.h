/* resident and dirty memory of this process, in MB */
#ifndef PERF_MEM_H
#define PERF_MEM_H
#include <stdio.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
static void mem_now(double *rss, double *dirty) {
    task_vm_info_data_t i; mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&i, &c);
    *rss = (double)i.resident_size / 1e6; *dirty = (double)i.phys_footprint / 1e6;
}
#else
#include <unistd.h>
static void mem_now(double *rss, double *dirty) {
    long total = 0, res = 0, shr = 0; FILE *f = fopen("/proc/self/statm", "r");
    if (f) { if (fscanf(f, "%ld %ld %ld", &total, &res, &shr) != 3) res = shr = 0; fclose(f); }
    double pg = (double)sysconf(_SC_PAGESIZE) / 1e6;
    *rss = (double)res * pg; *dirty = (double)(res - shr) * pg;
}
#endif
#endif
