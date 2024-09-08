#define DEFAULT_CHAR 'a'
#define GET_SECOND_ARG(a1, a2, ...) a2
#define GET_FLAG(...) GET_SECOND_ARG(0, ##__VA_ARGS__, DEFAULT_CHAR)