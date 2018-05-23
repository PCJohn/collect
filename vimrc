set nocp
set backspace=indent,eol,start
set smartindent
set tabstop=4
set shiftwidth=4
set expandtab
set nu

"Saving
map <c-o> :w<CR>
imap <c-o> <Esc>:w<CR>

"Exiting
map <c-x> :q<CR>
imap <c-x> <Esc>:q<CR>

"Select all
imap <c-a> <Esc>GVgg
map <c-a> GVgg

"New file
imap <c-n> <c-s><Esc>:enew
map <c-n> <c-s>:enew

"Yank
map <c-c> y
imap <c-c> <Esc>y

"Shortcut to yank a line
map <c-k> dd
imap <c-k> <Esc>dd

"Paste
map <c-v> p
imap <c-v> <Esc>p<Esc>i

"Find
map <c-f> /
imap <c-f> <Esc>/

"Undo
map <c-z> u
imap <c-z> <Esc>u<Esc>i

:syntax on
:fixdel
