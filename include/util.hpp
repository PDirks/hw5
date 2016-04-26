/*
 * Brendan Niebruegge
 * Peter Dirks
 * Homework 5
 * util.hpp
 * April 26, 2016
 */

// testing helpers
#ifndef assert
#define assert(e) if((e) != true){ \
                   fprintf(stderr,"%s,%d: assertion '%s' failed\n",__FILE__, __LINE__, #e); \
                   fflush(stderr); fflush(stdout); abort();}
#endif

// debug helpers
#ifndef debug_err
#define debug_err(e) \
    std::cerr << BRED << "[DEBUG] " << e << GREY << std::endl;
#endif
#ifndef debug_msg
#define debug_msg(e) \
    std::cout << GREEN << "[DEBUG] " << e << GREY << std::endl;
#endif

// colors
#ifndef BLACK
#define BLACK   "\033[0;30m"
#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define BROWN   "\033[0;33m"
#define BLUE    "\033[0;34m"
#define MAGENTA "\033[0;35m"
#define CYAN    "\033[0;36m"
#define GREY    "\033[0;37m"

#define BRED    "\033[1;31m"
#define BGREEN  "\033[1;32m"
#define BBLUE   "\033[1;34m"
#define BCYAN   "\033[1;36m"
#define BGREY   "\033[1;37m"
#endif

#flags
#define DEBUG               0