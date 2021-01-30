# ai_movie_edit

OpenCV + NLP project automatically finds the editing location of the video based on the scenario and tells you where to edit it.



<!--[if IE]><meta http-equiv="X-UA-Compatible" content="IE=5,IE=9" ><![endif]-->
<!DOCTYPE html>
<html>
<head>
<title>Untitled Diagram</title>
<meta charset="utf-8"/>
</head>
<body><div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;app.diagrams.net\&quot; modified=\&quot;2021-01-30T12:27:08.519Z\&quot; agent=\&quot;5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36\&quot; etag=\&quot;PEjvvYoOWF0mVhS8U2lv\&quot; version=\&quot;14.2.9\&quot; type=\&quot;github\&quot;&gt;&lt;diagram id=\&quot;C5RBs43oDa-KdzZeNtuy\&quot; name=\&quot;Page-1\&quot;&gt;7VpbV+M2EP41Pqd9gGPHkMtjbrBtKd0lu2V5VGzFFpEtIyu3/fU7siVfQ+IESJceeEik8ejimW++GYkY9jBYX3MU+X8zF1OjZbprwx4ZrVbPtOBTCjapoNvqpQKPEzcVWblgQn5gJTSVdEFcHJcUBWNUkKgsdFgYYkeUZIhztiqrzRgtrxohD9cEEwfRuvSeuMLXb9HJ5Z8w8Xy9stVW7xcgrazeJPaRy1YFkT027CFnTKStYD3EVNpO2+X+j809vZm3r//8Ej+hb4O/vt7+e5ZOdnXIkOwVOA7F607dSqdeIrpQ9lLvKjbagJwtQhfLSUzDHvgioNC0oPmIhdgoh6OFYCBiXPjMYyGiN4xFSm/GQqHULNnHoduXjoX+lDJnnoquCKVqDegp/S70YsHZPPOdnCBzhFSmaIrpADlzL9nokFHG4VHIQiyncgEM6l3yzY1z6aChbZUPYrbgDt6hZyuII+7hXfO1Uz25vwJOleeuMQuw4BtQ4JgiQZZlMCMVE16ml/sdGsr1B8DArsFgvBYcQTi2zDnerBh3jdYwCd2IOPC9iEnowfftzecdgJHOWvlE4EmEEqutgGLKICqCA95/4FEUx8q1ezx/mOeWmAu83mlr/bStAn6jKTDtrnL6sDQn+AXquDDfyDsXNe88AKN+xOlL4rRp/CkMaH83Dkc102dGQlFQYbNZDLxQ4mmto+CXraWX7lVwlXKLGlWBVrbz49F2WUPbLauBbb+/3jccQ7DhdzUu6TzIceeXujtaZ4vK3uYoSmoK0r3JxD4MzNbpwGxXwWw1AzN4Gm0KapFUiLfsRa9jb1/n2X1V9K2yPjTSHbxqZLWfy7IQJkmeZSuZZ2EnbRTIPJl+ZmnXvqqTvs+C6SLen2dL4JehdYUCQuXbf8J0iQVx0JZsjCjxQug4gGfMt0cVLClrAXvUzntfkyiGxHW6LG11Gqbp7lul6e5HmgaNNRHfc66E3oOmUWjntCk7G6PAoYVBGd++GdsGrP/lanS3/BHfDzqjb5O7p8lUn4WKbLtVrynbKqCa5zb8ldnpv6sm7E4F/qlZagS8f6LL05YllmaI4lG1TUVKOmEpyNpPC3koT0LlbKaIrl9kVgMygGSnkEGgwVAUxrVnif5w7yjzMTp66JwfMNTPiLr5mIA4nMVsJh25QUCDB4xFUQT2hL3KnOQx4ScnvxCzQ9ZH1FuEhfGNRxrjgdHvGD3LGA+N3siQtG0a474xsI2+KZ/2Loze5QEzukwsgl36Q+WYsxhzMpOi5PE0Y8MzJ6VDCSXuTX9rXdrZwEr79xyF0PLUN9WofB6ucULVcoVutK5Pcoed5DBuCh8nDOIwL4QhUkQC5MmMY6Iw0SBB0iNhpu4yZxEAWwLrsPBcbwjiMd1TeZ8gTiNLi//n5/yLbsMKosp8x1QQWzOLWaO4CbCq0A6Zcu0LypD0cAx1GeKEpZ59mYOesV5Twzev0054m7LVyNvySPV0m5dUjsSprL2Ltjp9kVMpNl6x6jHrVc/zZ8y6lwtuvNwRKsdWOvq4ZlaKj1YFHU2rmKxq0ZVYtRx6vSpmV41ZAJ8K5YAtiWRoh5LofQRy43tR+60iuX5r/c/0ESeX1iMsoAFJTuZv+RK4fNJOb7SPPUyfysR2u6GJX+NMu9XE9SuLj8vAwy4Dj2TuX+HyLoOVLpDMI1m3OpFdneiNbgH1hpveAtpl/RffAm4NqU4tpD6uid51BdX0X77WL31PVA0FvZWDY706Uacy0dEVFnTzH1uk6vkvVuzxTw==&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body>
</html>
