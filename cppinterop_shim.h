#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles from CXCppInterOp.h */
typedef void* CXInterpreter;
typedef void* CXScope;
typedef void* CXValue;
typedef void* CXObject;   /* actually void* in the C API */
typedef void* CXString;   /* use helper to get C string if you need it */

/* Core entrypoints you likely need (subset — add more later) */
CXInterpreter clang_createInterpreter(const char* const* argv, int argc);
void          clang_Interpreter_dispose(CXInterpreter I);

int           clang_Interpreter_declare(CXInterpreter I, const char* code, _Bool silent);
int           clang_Interpreter_process(CXInterpreter I, const char* code, _Bool silent);

CXScope       clang_lookupDeclName(const char* name, CXScope ctx); /* if present in your build */
CXScope       clang_instantiateTemplate(CXScope scope, const char* member, const char* tmpl_args);

void          clang_invoke(CXScope func, void* result, void** args, unsigned nArgs);

/* No default args here: provide explicit values from Lisp */
CXObject      clang_construct(CXScope scope, void* arena, size_t count);
_Bool         clang_destruct (CXObject This, CXScope S, _Bool withFree, size_t nary);

#ifdef __cplusplus
}
#endif
