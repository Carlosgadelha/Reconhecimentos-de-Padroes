% autores: 
%         José Carlos Silva Gadelha -> matrícula: 389110
%         André Luís Marques Rodrigues -> matrícula: 374866


aerogerador = importdata('aerogerador.dat');

x = aerogerador(:,1);
y = aerogerador(:,2);

tamanho_x = length(x); %quantidade de dados
tamanho_y = length(y); % quantidade de dados

lambda= 0.01; % valor fornecido na questão

% plotando os dados
subplot(3,2,1);
plot(x,y,'b*'); 
title("Aerogerador");
xlabel("Velocidade");
ylabel("Potencia");


for controlador = 1:5
  switch controlador
    case 1     %Calculando para Grau 1
        
        grau = controlador;
        I=eye(grau + 1);%matriz identidade
        X=[ones(tamanho_x,1),x ];
        Beta=((X'*X+lambda*I)^(-1)*X'*y); %Regularização de Tikhov
        
        y_aprox=(X*Beta);
        subplot(3,2,2);
        plot(x,y,'b*',x,y_aprox,'y-');
        title('Grau 1')
        xlabel('Velocidade')
        ylabel('Potência')
        
        
        e=y-y_aprox;      %calculo do erro
        SQe=sum((e).^2);
        S=sum((y-mean(y)).^2);
        R2_aj = 1-((SQe/(tamanho_x-grau))/(S/(tamanho_x-1)));
        fprintf("R2_aj ajustado para o polinomio de grau 1: %f \n", R2_aj);
        
     case 2     %Calculando para Grau 2
        
        grau = controlador;
        I=eye(grau + 1);%matriz identidade
        X=[ones(tamanho_x,1),x,(x.^2)];
        Beta=((X'*X+lambda*I)^(-1)*X'*y); %Regularização de Tikhov
        
        y_aprox=(X*Beta);
        subplot(3,2,3);
        plot(x,y,'b*',x,y_aprox,'m-');
        title('Grau 2')
        xlabel('Velocidade')
        ylabel('Potência')
        
       
        e=y-y_aprox;      %calculo do erro
        SQe=sum((e).^2);
        S=sum((y-mean(y)).^2);
        R2_aj = 1-((SQe/(tamanho_x-grau))/(S/(tamanho_x-1)));
        fprintf("R2_aj ajustado para o polinomio de grau 2: %f \n", R2_aj);
        
     case 3     %Calculando para Grau 2
        
        grau = controlador;
        I=eye(grau + 1);%matriz identidade
        X=[ones(tamanho_x,1),x,(x.^2),(x.^3)];
        Beta=((X'*X+lambda*I)^(-1)*X'*y); %Regularização de Tikhov
        
        y_aprox=(X*Beta);
        subplot(3,2,4);
        plot(x,y,'b*',x,y_aprox,'g-');
        title('Grau 3')
        xlabel('Velocidade')
        ylabel('Potência')
        
        
        e=y-y_aprox;      %calculo do erro
        SQe=sum((e).^2);
        S=sum((y-mean(y)).^2);
        R2_aj = 1-((SQe/(tamanho_x-grau))/(S/(tamanho_x-1)));
        fprintf("R2_aj ajustado para o polinomio de grau 3: %f \n", R2_aj);
      
     case 4     %Calculando para Grau 2
        
        grau = controlador;
        I=eye(grau + 1);%matriz identidade
        X=[ones(tamanho_x,1),x,(x.^2),(x.^3),(x.^4)];
        Beta=((X'*X+lambda*I)^(-1)*X'*y); %Regularização de Tikhov
        
        y_aprox=(X*Beta);
        subplot(3,2,5);
        plot(x,y,'b*',x,y_aprox,'k-');
        title('Grau 4')
        xlabel('Velocidade')
        ylabel('Potência')
        
        
        e=y-y_aprox;      %calculo do erro
        SQe=sum((e).^2);
        S=sum((y-mean(y)).^2);
        R2_aj = 1-((SQe/(tamanho_x-grau))/(S/(tamanho_x-1)));
        fprintf("R2_aj ajustado para o polinomio de grau 4: %f \n", R2_aj);
        
        
      case 5     %Calculando para Grau 2
        
        grau = controlador;
        I=eye(grau + 1);%matriz identidade
        X=[ones(tamanho_x,1),x,(x.^2),(x.^3),(x.^4),(x.^5)];
        Beta=((X'*X+lambda*I)^(-1)*X'*y); %Regularização de Tikhov
        
        y_aprox=(X*Beta);
        subplot(3,2,6);
        plot(x,y,'b*',x,y_aprox,'r-');
        title('Grau 5')
        xlabel('Velocidade')
        ylabel('Potência')
        
        
        e= y-y_aprox;  %erro
        SQe=sum((e).^2);
        S=sum((y-mean(y)).^2);
        R2_aj = 1-((SQe/(tamanho_x-grau))/(S/(tamanho_x-1)));
        fprintf("R2_aj ajustado para o polinomio de grau 5: %f \n", R2_aj);
      
  end
end

fprintf(" Ao analisar os valores de R2_aj notamos que um polinomio de grau 4 ja consegue representa bem os dados, \n pois um de grau 5 obteve o mesmo valor, ou seja omelhr valor de k seria 4 ");



























