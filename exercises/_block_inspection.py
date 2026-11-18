"""测评按实际子层身份定位Block角色，不修改或代替学习者的计算。"""


def learner_type(scope, name):
    if name in scope:
        return scope[name]
    # 本地按模块导入时，内部类在实际个人子层的定义模块中。
    for key in ('MLPBranch', 'MultiHeadAttention'):
        cls = scope.get(key)
        value = getattr(getattr(cls, '__init__', None), '__globals__', {}).get(name)
        if isinstance(value, type):
            return value
    return None


def block_parts(obj, scope, require_branch=True):
    def unique(cls, excluded=()):
        found = {id(v): v for v in vars(obj).values()
                 if cls is not None and type(v) is cls and all(v is not e for e in excluded)}
        if len(found) != 1:
            raise AssertionError('Block须持有唯一的当前个人 '+getattr(cls, '__name__', '子层')+' 实例')
        return next(iter(found.values()))
    branch_cls = scope.get('MLPBranch')
    branches = {id(v): v for v in vars(obj).values() if branch_cls is not None and type(v) is branch_cls}
    if branches or require_branch:
        branch = unique(branch_cls)
        norm2, mlp = branch.norm, branch.mlp
        norm1 = unique(learner_type(scope, 'RMSNorm'), (norm2,))
        attn = unique(scope.get('MultiHeadAttention'))
        return norm1, attn, branch, norm2, mlp
    # 可选历史Block仍使用原始结构，不增加主线别名要求。
    return obj.norm1, obj.attn, None, obj.norm2, obj.mlp
