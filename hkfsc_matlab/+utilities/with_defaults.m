function out = with_defaults(in, defaults)
%WITH_DEFAULTS Fill absent fields without changing explicitly supplied values.
out = defaults;
if nargin < 1 || isempty(in), return; end
names = fieldnames(in);
for i = 1:numel(names)
    key = names{i};
    if isstruct(in.(key)) && isfield(out, key) && isstruct(out.(key))
        out.(key) = utilities.with_defaults(in.(key), out.(key));
    else
        out.(key) = in.(key);
    end
end
end

