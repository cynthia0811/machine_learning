import fool
import jieba


def main():
    text_list =["工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "一男子扬言要整成都教授，成都教授纷纷搬家",
    "程维谈滴滴停掉与点评接口：没必要假惺惺合作",
    "这个感冒药出事了 全国停止销售并召回",
    "长虹（CHANGHONG）55D3C 55英寸 32核4K超高清HDR超薄曲面人工智能液晶电视机（黑色） ",
    "绿联（UGREEN）HDMI转VGA线转换器带音频口 高清视频转接头适配器 电脑盒子连接投影仪电视显示器线 黑 40248 "] 

    for text in text_list:
        print("fool:",fool.cut(text))
        print("jieba:","/ ".join(jieba.cut(text)))
  


if __name__ == '__main__':
    main()