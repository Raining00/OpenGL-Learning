#pragma once

#include <iostream>
#include <string>

#define PRINT_RED(str) std::cout<<"\033[31m"<<str<<"\033[0m"<<std::endl;
#define PRINT_GREEN(str) std::cout<<"\033[32m"<<str<<"\033[0m"<<std::endl;
#define PRINT_YELLOW(str) std::cout<<"\033[33m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BLUE(str) std::cout<<"\033[34m"<<str<<"\033[0m"<<std::endl;
#define PRINT_PURPLE(str) std::cout<<"\033[35m"<<str<<"\033[0m"<<std::endl;
#define PRINT_CYAN(str) std::cout<<"\033[36m"<<str<<"\033[0m"<<std::endl;
#define PRINT_WHITE(str) std::cout<<"\033[37m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BLACK(str) std::cout<<"\033[30m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED(str) std::cout<<"\033[1;31m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN(str) std::cout<<"\033[1;32m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW(str) std::cout<<"\033[1;33m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE(str) std::cout<<"\033[1;34m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE(str) std::cout<<"\033[1;35m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN(str) std::cout<<"\033[1;36m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE(str) std::cout<<"\033[1;37m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLACK(str) std::cout<<"\033[1;30m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED_BBLACK(str) std::cout<<"\033[1;31;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN_BBLACK(str) std::cout<<"\033[1;32;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW_BBLACK(str) std::cout<<"\033[1;33;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE_BBLACK(str) std::cout<<"\033[1;34;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE_BBLACK(str) std::cout<<"\033[1;35;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN_BBLACK(str) std::cout<<"\033[1;36;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE_BBLACK(str) std::cout<<"\033[1;37;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLACK_BBLACK(str) std::cout<<"\033[1;30;40m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED_BWHITE(str) std::cout<<"\033[1;31;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN_BWHITE(str) std::cout<<"\033[1;32;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW_BWHITE(str) std::cout<<"\033[1;33;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE_BWHITE(str) std::cout<<"\033[1;34;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE_BWHITE(str) std::cout<<"\033[1;35;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN_BWHITE(str) std::cout<<"\033[1;36;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE_BWHITE(str) std::cout<<"\033[1;37;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLACK_BWHITE(str) std::cout<<"\033[1;30;47m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED_BCYAN(str) std::cout<<"\033[1;31;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN_BCYAN(str) std::cout<<"\033[1;32;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW_BCYAN(str) std::cout<<"\033[1;33;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE_BCYAN(str) std::cout<<"\033[1;34;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE_BCYAN(str) std::cout<<"\033[1;35;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN_BCYAN(str) std::cout<<"\033[1;36;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE_BCYAN(str) std::cout<<"\033[1;37;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLACK_BCYAN(str) std::cout<<"\033[1;30;46m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED_BBLUE(str) std::cout<<"\033[1;31;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN_BBLUE(str) std::cout<<"\033[1;32;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW_BBLUE(str) std::cout<<"\033[1;33;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE_BBLUE(str) std::cout<<"\033[1;34;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE_BBLUE(str) std::cout<<"\033[1;35;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN_BBLUE(str) std::cout<<"\033[1;36;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE_BBLUE(str) std::cout<<"\033[1;37;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLACK_BBLUE(str) std::cout<<"\033[1;30;44m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED_BYELLOW(str) std::cout<<"\033[1;31;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN_BYELLOW(str) std::cout<<"\033[1;32;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW_BYELLOW(str) std::cout<<"\033[1;33;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE_BYELLOW(str) std::cout<<"\033[1;34;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE_BYELLOW(str) std::cout<<"\033[1;35;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN_BYELLOW(str) std::cout<<"\033[1;36;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE_BYELLOW(str) std::cout<<"\033[1;37;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLACK_BYELLOW(str) std::cout<<"\033[1;30;43m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BRED_BPURPLE(str) std::cout<<"\033[1;31;45m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BGREEN_BPURPLE(str) std::cout<<"\033[1;32;45m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BYELLOW_BPURPLE(str) std::cout<<"\033[1;33;45m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BBLUE_BPURPLE(str) std::cout<<"\033[1;34;45m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BPURPLE_BPURPLE(str) std::cout<<"\033[1;35;45m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BCYAN_BPURPLE(str) std::cout<<"\033[1;36;45m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BWHITE_BPURPLE(str) std::cout<<"\033[1;37;45m"<<str<<"\033[0m"<<std::endl;

#define PRINT_RED_BLINK(str) std::cout<<"\033[31;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_GREEN_BLINK(str) std::cout<<"\033[32;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_YELLOW_BLINK(str) std::cout<<"\033[33;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BLUE_BLINK(str) std::cout<<"\033[34;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_PURPLE_BLINK(str) std::cout<<"\033[35;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_CYAN_BLINK(str) std::cout<<"\033[36;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_WHITE_BLINK(str) std::cout<<"\033[37;5m"<<str<<"\033[0m"<<std::endl;
#define PRINT_BLACK_BLINK(str) std::cout<<"\033[30;5m"<<str<<"\033[0m"<<std::endl;

// special use
// error message is red and blink   
#define PRINT_ERROR(str) std::cout<<"\033[31;5m"<<str<<"\033[0m"<<std::endl;
// warning message is yellow and blink
#define PRINT_WARNING(str) std::cout<<"\033[33;5m"<<str<<"\033[0m"<<std::endl;
// info message is green and blink
#define PRINT_INFO(str) std::cout<<"\033[32;5m"<<str<<"\033[0m"<<std::endl;
// ok message is cyan and blink
#define PRINT_OK(str) std::cout<<"\033[36;5m"<<str<<"\033[0m"<<std::endl;
// debug message is blue and blink
#define PRINT_DEBUG(str) std::cout<<"\033[34;5m"<<str<<"\033[0m"<<std::endl;
// trace message is purple and blink
#define PRINT_TRACE(str) std::cout<<"\033[35;5m"<<str<<"\033[0m"<<std::endl;
// fatal message is white and blink
#define PRINT_FATAL(str) std::cout<<"\033[37;5m"<<str<<"\033[0m"<<std::endl;
// unknown message is black and blink
#define PRINT_UNKNOWN(str) std::cout<<"\033[30;5m"<<str<<"\033[0m"<<std::endl;