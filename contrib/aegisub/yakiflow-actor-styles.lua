script_name = "Fill actor styles"
script_description = "Give each actor's Default lines the style of that actor's first non-Default line"
script_author = "YakiFlow contributors"
script_version = "1.0"

local function apply(subs)
    -- First pass: remember the first non-Default style each actor uses.
    local first_style = {}
    for i = 1, #subs do
        local l = subs[i]
        if l.class == "dialogue" and l.actor ~= ""
           and l.style ~= "Default" and not first_style[l.actor] then
            first_style[l.actor] = l.style
        end
    end

    -- Second pass: only touch Default lines.
    local changed = 0
    for i = 1, #subs do
        local l = subs[i]
        if l.class == "dialogue" and l.style == "Default"
           and first_style[l.actor] then
            l.style = first_style[l.actor]
            subs[i] = l
            changed = changed + 1
        end
    end

    aegisub.log(string.format("Changed %d lines\n", changed))
    aegisub.set_undo_point(script_name)
end

aegisub.register_macro(script_name, script_description, apply)
