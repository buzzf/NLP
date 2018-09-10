#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: zzf
# @Date:   2018-08-31 19:41:04
# @Last Modified by:   zzf
# @Last Modified time: 2018-09-01 19:39:21




with open('./data/poems.txt', "r", encoding='utf-8', ) as fr:
    with open('./data/poems_cut.txt', 'w', encoding='utf-8') as fw:
        for line in fr.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 160:
                    continue
                content = 'G' + content + 'E'
                words = [word for word in content]
                string = ' '.join(words)
                fw.write(string)
                fw.write('\n')
                
            except ValueError as e:
                pass

