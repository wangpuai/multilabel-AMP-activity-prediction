function Average_Precision = Metric_AveragePrecision(Outputs,test_target)
%Computing the average precision
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

    [num_class,num_instance] = size(Outputs);
    temp_Outputs = [];
    temp_test_target = [];
    for i = 1:num_instance
        temp = test_target(:,i);   %��i�����������ı��������1-(-1)����
        if((sum(temp)~=num_class)&(sum(temp)~=-num_class))  %��ȫΪ1�����߲�ȫΪ-1
            temp_Outputs = [temp_Outputs, Outputs(:,i)];
            temp_test_target = [temp_test_target, temp];
        end
    end
    Outputs = temp_Outputs;    %������ȫΪ1��ȫΪ-1�Ĳ�������ȥ��
    test_target = temp_test_target;     
    [num_class,num_instance] = size(Outputs);
    
    Label = cell(num_instance,1);      %ÿ�������������еı�ǩ����
    not_Label = cell(num_instance,1);  %ÿ���������������еı�ǩ����
    Label_size = zeros(1,num_instance);  %ÿ�������������еı�ǩ��
    for i = 1:num_instance
        temp = test_target(:,i);
        Label_size(1,i) = sum(temp==ones(num_class,1));
        for j = 1:num_class
            if(temp(j)==1)
                Label{i,1} = [Label{i,1},j];
            else
                not_Label{i,1} = [not_Label{i,1},j];
            end
        end
    end
    
    aveprec = 0;
    for i = 1:num_instance
        temp = Outputs(:,i);
        [~,index] = sort(temp);  %���յ�i���������и�����ǩ�ĸ��ʴ�С���� �������б�ǩ��������index��
        indicator = zeros(1,num_class);
        for m = 1:Label_size(i)
            [~,loc] = ismember(Label{i,1}(m),index);  %��i�����������ı�ǩ���е�m����ǩ��Ӧindex�е�λ��
            indicator(1,loc) = 1;   %indicatorΪ0-1��������[0 1 1 0]��ʵ���Ͼ��ǽ���i�������ĶԸ���ǩ�ĸ��ʴ�С�������У���������ĳ��ǩ��Ӧ�ĸ�����1��������0
        end
        summary = 0;
        for m = 1:Label_size(i)
            [~,loc] = ismember(Label{i,1}(m),index);
            summary = summary + sum(indicator(loc:num_class))/(num_class-loc+1);  %�ۼӵ�i���������е�ÿ����ǩ����Ӧ�ľ��ȣ���[0 1 1 0]��summary = 1/2 + 2/3
        end
%         ap_binary(i) = summary/Label_size(i);
        aveprec = aveprec + summary/Label_size(i);   %��i��������ƽ������
    end
    Average_Precision = aveprec/num_instance;   %����������ƽ������
    