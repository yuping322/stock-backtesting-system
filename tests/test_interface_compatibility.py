"""
data_akshare.py 和 data.py 接口参数对比测试
验证所有接口的参数完全一致
"""

import inspect
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# 导入两个模块
try:
    import data
    import data_akshare
    print("✅ 成功导入data和data_akshare模块")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def get_function_signature(func):
    """获取函数的完整签名"""
    sig = inspect.signature(func)
    params = []
    for name, param in sig.parameters.items():
        param_str = name
        if param.annotation != inspect.Parameter.empty:
            param_str += f": {param.annotation}"
        if param.default != inspect.Parameter.empty:
            param_str += f" = {param.default}"
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            param_str = f"**{name}"
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            param_str = f"*{name}"
        params.append(param_str)
    
    return_str = ""
    if sig.return_annotation != inspect.Signature.empty:
        return_str = f" -> {sig.return_annotation}"
    
    return f"({', '.join(params)}){return_str}"


def compare_interfaces():
    """对比两个模块的接口"""
    
    # 需要对比的接口列表
    interfaces = [
        'load_new_stocks',
        'load_oss_stocks',
        'read_factor_data',
        'read_factor_data_loal',
        'factor_for_al',
        'get_balance',
        'get_income',
        'get_cashflow',
        'get_valuation',
        'get_history_fundamentals',
        'get_index_stocks',
        'get_index_daily',
        'load_bt_stocks',
        'load_bt_pricing',
        'get_trading_dates',
        '_normalize_date_arg',
        '_normalize_code_arg',
        '_ensure_exchange_prefix',
        '_ensure_exchange_suffix',
        '_add_prefix',
        '_parse_date',
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("接口参数对比测试")
    print("="*80)
    
    for interface_name in interfaces:
        if not hasattr(data, interface_name):
            print(f"⚠️  data.py 中没有 {interface_name}")
            continue
        if not hasattr(data_akshare, interface_name):
            print(f"⚠️  data_akshare.py 中没有 {interface_name}")
            continue
        
        func_data = getattr(data, interface_name)
        func_akshare = getattr(data_akshare, interface_name)
        
        sig_data = get_function_signature(func_data)
        sig_akshare = get_function_signature(func_akshare)
        
        is_match = sig_data == sig_akshare
        
        results.append({
            'name': interface_name,
            'data_signature': sig_data,
            'akshare_signature': sig_akshare,
            'match': is_match
        })
        
        status = "✅" if is_match else "❌"
        print(f"\n{status} {interface_name}")
        print(f"  data.py:        {sig_data}")
        print(f"  data_akshare.py: {sig_akshare}")
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    matched = sum(1 for r in results if r['match'])
    total = len(results)
    
    print(f"✅ 匹配: {matched}/{total}")
    print(f"❌ 不匹配: {total - matched}/{total}")
    print(f"📊 匹配率: {matched/total*100:.1f}%")
    
    # 显示不匹配的接口
    mismatches = [r for r in results if not r['match']]
    if mismatches:
        print("\n不匹配的接口:")
        for r in mismatches:
            print(f"\n❌ {r['name']}")
            print(f"  data.py:        {r['data_signature']}")
            print(f"  data_akshare.py: {r['akshare_signature']}")
    
    return matched == total


def test_function_call():
    """测试函数调用是否正常工作"""
    print("\n" + "="*80)
    print("函数调用测试")
    print("="*80)
    
    test_cases = [
        {
            'func': 'get_trading_dates',
            'args': ('2024-01-01', '2024-01-05'),
            'kwargs': {'as_str': False}
        },
        {
            'func': '_normalize_code_arg',
            'args': ('000001',),
            'kwargs': {}
        },
        {
            'func': '_ensure_exchange_prefix',
            'args': ('000001',),
            'kwargs': {}
        },
        {
            'func': '_ensure_exchange_suffix',
            'args': ('000001',),
            'kwargs': {}
        },
    ]
    
    for test_case in test_cases:
        func_name = test_case['func']
        args = test_case['args']
        kwargs = test_case['kwargs']
        
        try:
            func_data = getattr(data, func_name)
            func_akshare = getattr(data_akshare, func_name)
            
            result_data = func_data(*args, **kwargs)
            result_akshare = func_akshare(*args, **kwargs)
            
            # 简单比较（忽略类型细节）
            if str(result_data) == str(result_akshare):
                print(f"✅ {func_name}({args}, {kwargs}) - 结果一致")
            else:
                print(f"⚠️  {func_name}({args}, {kwargs}) - 结果不同但签名一致")
                print(f"   data: {result_data}")
                print(f"   akshare: {result_akshare}")
        except Exception as e:
            print(f"❌ {func_name} 测试失败: {e}")


if __name__ == "__main__":
    # 对比接口签名
    all_match = compare_interfaces()
    
    # 测试函数调用
    test_function_call()
    
    # 最终结果
    print("\n" + "="*80)
    if all_match:
        print("🎉 所有接口参数完全一致！")
    else:
        print("⚠️  存在参数不一致的接口，请检查并修复")
    print("="*80)

