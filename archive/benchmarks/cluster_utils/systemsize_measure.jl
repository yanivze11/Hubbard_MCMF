using Printf
using Dates

# ═══════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════

"""
    folder_size(path::AbstractString) -> Int

Return the total size of all files in `path` (recursively) in bytes.
"""
function folder_size(path::AbstractString)
    total = 0
    for (root, dirs, files) in walkdir(path)
        for f in files
            fp = joinpath(root, f)
            try
                total += filesize(fp)
            catch
            end
        end
    end
    return total
end

"""
    format_size(bytes::Integer) -> String

Human-readable size string (B / KB / MB / GB / TB).
"""
function format_size(bytes::Integer)
    if bytes < 1024
        return @sprintf("%d B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.2f KB", bytes / 1024)
    elseif bytes < 1024^3
        return @sprintf("%.2f MB", bytes / 1024^2)
    elseif bytes < 1024^4
        return @sprintf("%.4f GB", bytes / 1024^3)
    else
        return @sprintf("%.4f TB", bytes / 1024^4)
    end
end

# ═══════════════════════════════════════════════════════════
#  Filesystem tree structure
# ═══════════════════════════════════════════════════════════

"""
    filesystem_tree(root; io=stdout, max_depth=typemax(Int), show_files=true, show_sizes=true)

Print the directory tree under `root` with optional file sizes.
Uses box-drawing characters for a nice tree view.

# Arguments
- `root`: path to the root directory.
- `io`: IO stream to write to (default: stdout).
- `max_depth`: how many levels deep to recurse (default: unlimited).
- `show_files`: whether to list individual files (default: true).
- `show_sizes`: whether to show sizes next to each entry (default: true).
"""
function filesystem_tree(root::AbstractString;
                         io::IO=stdout,
                         max_depth::Int=typemax(Int),
                         show_files::Bool=true,
                         show_sizes::Bool=true)
    root = abspath(root)
    if !isdir(root)
        error("Not a directory: $root")
    end

    # Print root
    root_sz = show_sizes ? "  [$(format_size(folder_size(root)))]" : ""
    println(io, root, root_sz)

    function _print_tree(dir, prefix, depth)
        depth >= max_depth && return

        entries = readdir(dir; join=false)
        # Separate dirs and files, sort each alphabetically
        dirs_here  = filter(e -> isdir(joinpath(dir, e)), entries)
        files_here = filter(e -> isfile(joinpath(dir, e)), entries)
        sort!(dirs_here)
        sort!(files_here)

        items = show_files ? vcat(dirs_here, files_here) : dirs_here
        n = length(items)

        for (i, name) in enumerate(items)
            full = joinpath(dir, name)
            is_last = (i == n)
            connector = is_last ? "└── " : "├── "
            child_prefix = is_last ? "    " : "│   "

            if isdir(full)
                sz_str = show_sizes ? "  [$(format_size(folder_size(full)))]" : ""
                println(io, prefix, connector, name, "/", sz_str)
                _print_tree(full, prefix * child_prefix, depth + 1)
            else
                sz_str = show_sizes ? "  [$(format_size(filesize(full)))]" : ""
                println(io, prefix, connector, name, sz_str)
            end
        end
    end

    _print_tree(root, "", 0)
end

# ═══════════════════════════════════════════════════════════
#  Folder-size report (table format)
# ═══════════════════════════════════════════════════════════

"""
    report_folder_sizes(root; io=stdout, depth=1, sort_by_size=true)

Print a table of subfolder sizes (in GB) under `root`.

- `depth`: how many levels of subfolders to report (1 = immediate children).
- `sort_by_size`: if `true`, results are printed largest-first.
"""
function report_folder_sizes(root::AbstractString;
                             io::IO=stdout,
                             depth::Int=1,
                             sort_by_size::Bool=true)
    root = abspath(root)
    if !isdir(root)
        error("Not a directory: $root")
    end

    entries = Tuple{String, Int}[]

    function _walk(dir, current_depth)
        for item in readdir(dir; join=true)
            if isdir(item)
                sz = folder_size(item)
                rel = relpath(item, root)
                push!(entries, (rel, sz))
                if current_depth < depth
                    _walk(item, current_depth + 1)
                end
            end
        end
    end

    _walk(root, 1)

    if sort_by_size
        sort!(entries; by=x -> x[2], rev=true)
    end

    println(io, "="^70)
    @printf(io, "  %-50s  %14s\n", "Folder", "Size")
    println(io, "="^70)
    for (name, sz) in entries
        @printf(io, "  %-50s  %14s\n", name, format_size(sz))
    end
    println(io, "-"^70)
    total_sz = folder_size(root)
    @printf(io, "  %-50s  %14s\n", "TOTAL", format_size(total_sz))
    println(io, "="^70)
end

# ═══════════════════════════════════════════════════════════
#  Full filesystem report (saved to file)
# ═══════════════════════════════════════════════════════════

"""
    generate_filesystem_report(root; outfile="filesystem_report.txt",
                               tree_depth=typemax(Int), table_depth=2,
                               show_files=true)

Generate a comprehensive filesystem report and save it to `outfile`.
The report contains:
1. Summary (timestamp, root path, total size, file/folder counts)
2. A size table of subfolders
3. The full directory tree with sizes

# Arguments
- `root`: the directory to scan.
- `outfile`: output filename (default: `"filesystem_report.txt"`).
- `tree_depth`: depth for the tree section (default: unlimited).
- `table_depth`: depth for the size table (default: 2).
- `show_files`: include individual files in the tree (default: true).
"""
function generate_filesystem_report(root::AbstractString;
                                    outfile::AbstractString="filesystem_report.txt",
                                    tree_depth::Int=1,
                                    table_depth::Int=2,
                                    show_files::Bool=true)
    root = abspath(root)
    if !isdir(root)
        error("Not a directory: $root")
    end

    # Count files and folders
    n_files = 0
    n_dirs  = 0
    for (r, ds, fs) in walkdir(root)
        n_dirs  += length(ds)
        n_files += length(fs)
    end

    total_bytes = folder_size(root)

    open(outfile, "w") do io
        # ── Header ──
        println(io, "╔", "═"^68, "╗")
        println(io, "║", lpad("FILESYSTEM SIZE REPORT", 44), " "^24, "║")
        println(io, "╚", "═"^68, "╝")
        println(io)

        # ── Summary ──
        println(io, "Generated : ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io, "Root      : ", root)
        println(io, "Total size: ", format_size(total_bytes))
        println(io, "Folders   : ", n_dirs)
        println(io, "Files     : ", n_files)
        println(io)

        # ── Size table ──
        println(io, "┌", "─"^68, "┐")
        println(io, "│  FOLDER SIZE TABLE (depth=$table_depth)", " "^(68 - 39 - length(string(table_depth))), "│")
        println(io, "└", "─"^68, "┘")
        println(io)
        report_folder_sizes(root; io=io, depth=table_depth, sort_by_size=true)
        println(io)

        # ── Tree ──
        println(io, "┌", "─"^68, "┐")
        println(io, "│  DIRECTORY TREE", " "^51, "│")
        println(io, "└", "─"^68, "┘")
        println(io)
        filesystem_tree(root; io=io, max_depth=tree_depth, show_files=show_files, show_sizes=true)
        println(io)

        println(io, "─"^70)
        println(io, "End of report.")
    end

    println("Report saved to: $outfile")
    println("  Total size: $(format_size(total_bytes))  |  $n_dirs folders  |  $n_files files")
end

# ═══════════════════════════════════════════════════════════
#  Main entry point (for SLURM / command-line execution)
# ═══════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    # Default: scan current directory, or pass a path as first argument
    scan_root = length(ARGS) >= 1 ? ARGS[1] : "."
    outfile   = length(ARGS) >= 2 ? ARGS[2] : "filesystem_report.txt"

    println("Scanning: $(abspath(scan_root))")
    println("Output : $outfile")
    println()

    generate_filesystem_report(scan_root; outfile=outfile)
end
